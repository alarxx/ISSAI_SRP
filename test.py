"""
SPDX-License-Identifier: MPL-2.0
--------------------------------
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file,
You can obtain one at https://mozilla.org/MPL/2.0/.

This file is part of the ISSAI Summer Research Project.

Most of the code is taken from tutorial below:
https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html

Provided “as is”, without warranty of any kind.

Copyright © 2025 Alar Akilbekov. All rights reserved.

Third party copyrights are property of their respective owners.
"""

import os
import platform
import sys
import tempfile
import datetime
import dotenv 
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
dotenv.load_dotenv()  # загрузит переменные из .env

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# utils
from utils.data.CustomImageDataset import CustomImageDataset
from utils import devices
from utils.visualizer import visualize_image_dataset, plot4
from utils import saver
# from utils.trainer import Trainer
from utils.model.MLP import MLP
# from utils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# ---

CHANNELS, HEIGHT, WIDTH = 3, 28, 28
BATCH_SIZE = 64

# ---

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost' # on cluster of machines should be explicit IP 
    os.environ['MASTER_PORT'] = '12480' # ports range 2^16: [0, 65535], better choose port>1024
    # https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html#constructing-the-process-group
    torch.cuda.set_device(rank) # sets the default GPU for each process
    # initialize process group
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        # initialize the process group
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=10) # Default is 10 minutes for NCCL and 30 minutes for other backends
            )

def cleanup():
    dist.destroy_process_group()


def prepare_dataset():
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),  # например
        transforms.ToTensor(), # tensor(C, H, W) in RGB
    ])

    dataset = CustomImageDataset(root_dir="dataset/CMNIST", transform=transform)

    # Как разделить на Train, Val, Test
    # validation_data = K-Fold Cross Validation on training_data
    training_data, test_data = dataset.train_test_split(test_size=0.25, random_state=42)
    
    # param dir_path, не храни пути, возвращай изображения по индексам восстанавливая путь к файлу

    # Create data loaders.
    # Data Loader wraps an iterable over dataset
    train_dataloader = DataLoader(
        dataset=training_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # NEW: False because of DistributedSampler below
        # Но почему shuffe нужно отключать, как работает DistributedSampler, как он обеспечивает non overlapping batches?
        # if shuffle and sampler is not None: throw error - лучше ошибка чем баг? Или происходит просто двойной shuffle?
        # DistributedSampler работает без синхронизации между процессами, использует rank и world_size из global state
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(training_data)  # NEW chunk batches across GPUs without overlapping samples
    )
    
    test_dataloader = DataLoader(
        dataset=test_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    return train_dataloader, test_dataloader, dataset.classes


def ddp_train_epoch(rank, world_size, ddp_model, loss_fn, dataloader, optimizer, epoch):
    device = rank
    dataloader.sampler.set_epoch(epoch)
    ddp_model.train()
    size = len(dataloader.dataset) # количество всех примеров в датасете
    num_batches = len(dataloader) * world_size # количество батчей, i.e. dataset/batch_size
    train_loss, n_correct = torch.tensor(0., device=device), torch.tensor(0., device=device)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = ddp_model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_ = loss 
        n_correct_  = (pred.argmax(1) == y).type(torch.float).sum()
        # print("loss:", loss.dtype, "; correct:", correct.dtype)

        # Synchronize current loss and number of correctly predicted
        dist.all_reduce(train_loss_,    op=dist.ReduceOp.SUM)
        dist.all_reduce(n_correct_,     op=dist.ReduceOp.SUM)

        train_loss  += train_loss_
        n_correct   += n_correct_

        if rank == 0 and batch % 100 == 0:
            loss, current = train_loss_.item(), (batch + 1) * dataloader.batch_size 
            print(f"[GPU{rank}] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss  /= num_batches  # средний loss на эпохе
    n_correct   /= size         # accuracy

    return train_loss.item(), n_correct.item() # each process translates to CPU but once in epoch


def validate(rank, ddp_model, loss_fn, dataloader):
    device = rank
    ddp_model.eval()
    size = len(dataloader.dataset) # количество всех примеров в датасете
    num_batches = len(dataloader) # количество батчей, i.e. dataset/batch_size
    test_loss, n_correct = torch.tensor(0., device=device), torch.tensor(0., device=device)
    with torch.no_grad(): # temporarily changing global state - torch.is_grad_enabled()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = ddp_model(X)
            test_loss += loss_fn(pred, y)
            n_correct += (pred.argmax(1) == y).type(torch.float).sum()

        test_loss /= num_batches
        n_correct /= size
    print(f"Test Error: \n Accuracy: {(100*n_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss.item(), n_correct.item()


def demo(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    # rank and world_size are global variables
    print("dist.get_rank():", dist.get_rank())
    print("dist.get_world_size():", dist.get_world_size())

    train_loader, test_loader, classes = prepare_dataset()
    visualize_image_dataset(train_loader.dataset, classes)

    model = MLP(n_in=CHANNELS*HEIGHT*WIDTH, n_out=len(classes)).to(rank)

    # DDP synchronizes randomly initialized parameters
    # DDP вешает hook на параметры для синхронизации, когда .grad будут готовы
    ddp_model = DDP(model, device_ids=[rank]) # device_ids перемещает input на нужный device
    # ddp_model = DDP(model) # не указываем device_ids при Model Parallel

    # --- Saving START ---

    for name, p in ddp_model.module.named_parameters():
        print(f"[rank{rank}] param {name} is on {p.device}")

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # Сохраняя state_dict сохраняются и device-ы на котором находились тензоры, here cuda:0
        # saver.print_state_dict(model)
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
        # What if we call dist.barrier() here, so for rank0 will be twice, will we just stuck on rank0 process?

    # Use a barrier() to make sure that process 1 loads the model after process 0 saves it.
    # A process is blocked by a barrier until all processes have encountered a barrier, upon which the barrier is lifted for all.
    # IMO, in our case no need to synchronize by device_ids, но было бы нужно если бы на одном процессе 2 потока использовали разные видеокарты асинхронно
    dist.barrier(device_ids=[rank]) # точка синхронизации, задача в очереди GPU по типу all-reduce блокирует thread-GPU
    # Configure map_location properly
    map_location = {f'cuda:{0}': f'cuda:{rank}'} # map parameters to corresponding process-GPU-device
    # map_location = {
    #     'cuda:0': f'cuda:{rank * 2}',
    #     'cuda:1': f'cuda:{rank * 2 + 1}'
    # }
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True)
    )

    # --- Saving END ---

    """
        Я так понял все градиенты вычисляются и потом синхронизируются.
        А на каком девайсе делается step?
        Мы синхронизируем градиенты и потом каждый процесс сам делает step, потому что иначе потом пришлось бы синхронизировать модель
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(ddp_model.parameters(), lr=0.001)

    EPOCHS = 5
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(EPOCHS):
        print(f"epoch {epoch}")

        train_loss, train_acc = ddp_train_epoch(
            rank, world_size,
            ddp_model,
            loss_fn,
            train_loader,
            optimizer,
            epoch
        )

        if rank == 0:
            print("Hell")
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            val_loss, val_acc = validate(
                rank,
                ddp_model,
                loss_fn,
                test_loader
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

        dist.barrier(device_ids=[rank])

    # Not necessary to use a dist.barrier() to guard the file deletion below
    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def handler(rank, world_size):
    try:
        demo(rank, world_size)
    except Exception as e:
        print(f"Exception on [rank{rank}]:", e)   
    finally:
        dist.barrier()  # чтобы не убегал никто раньше
        cleanup()
        print(f"Finished running basic DDP example on rank {rank}.")


# В дочерних процессах mp.spawn __name__ != "__main__"
if __name__ == "__main__":

    print("PyTorch version:", torch.__version__)
    if(not devices.print_available_devices()):
        exit()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    # world_size = n_gpus//2 # works on odd number of GPUs
    world_size = n_gpus
    mp.spawn(handler,
             args=(world_size,),
             nprocs=world_size,
             join=True)
