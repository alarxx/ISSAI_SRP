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

# utils
# from utils.data.CustomImageDataset import CustomImageDataset
from utils import devices
# from utils.plot import plot4
from utils import saver
# from utils.trainer import Trainer
# from utils.model.MLP import MLP
# from utils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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
    os.environ['MASTER_PORT'] = '12355' # ports range 2^16: [0, 65535], better choose port>1024
    # https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html#constructing-the-process-group
    torch.cuda.set_device(rank) # sets the default GPU for each process
    # initialize the process group
    dist.init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=datetime.timedelta(minutes=10) # Default is 10 minutes for NCCL and 30 minutes for other backends
        )

def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)
        print(f"[forward] net1 on {self.net1.weight.device}")
        print(f"[forward] net2 on {self.net2.weight.device}")

    def forward(self, x):
        x = x.to(self.dev0)
        # print(f"[forward] input is on {x.device}")
        x = self.net1(x)
        x = self.relu(x)
        x = x.to(self.dev1)
        x = self.net2(x)
        return x


def demo(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    print(f"[rank{rank}] uses dev0={dev0}, dev1={dev1}")

    # create model and move it to GPU with id rank
    # model = ToyModel(dev0, dev1).to(rank)
    model = ToyModel(dev0, dev1) # не указываем device при Model Parallel

    # DDP synchronizes randomly initialized parameters
    # DDP вешает hook на параметры для синхронизации, когда .grad будут готовы
    # ddp_model = DDP(model, device_ids=[rank]) # device_ids перемещает input на нужный device
    ddp_model = DDP(model) # не указываем device_ids при Model Parallel

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
    dist.barrier(device_ids=[dev0, dev1]) # точка синхронизации, задача в очереди GPU по типу all-reduce блокирует thread-GPU
    # Configure map_location properly
    # map_location = {f'cuda:{0}': f'cuda:{rank}'} # map parameters to corresponding process-GPU-device
    map_location = {
        'cuda:0': f'cuda:{rank * 2}',
        'cuda:1': f'cuda:{rank * 2 + 1}'
    }
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True)
    )

    """
        Я так понял все градиенты вычисляются и потом синхронизируются.
        А на каком девайсе делается step?
        Мы синхронизируем градиенты и потом каждый процесс сам делает step, потому что иначе потом пришлось бы синхронизировать модель
    """
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for i in range(10000):
        print(f"epoch {i}")
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(dev1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


if __name__ == "__main__":

    if(not devices.print_available_devices()):
        exit()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    world_size = n_gpus//2 # works on odd number of GPUs
    mp.spawn(demo,
             args=(world_size,),
             nprocs=world_size,
             join=True)
