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
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank]) # synchronizes randomly initialized parameters

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # Сохраняя state_dict сохраняются и device-ы на котором находились тензоры, here cuda:0
        # saver.print_state_dict(model)
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process 0 saves it.
    dist.barrier(device_ids=[rank]) # точка синхронизации, задача в очереди GPU по типу all-reduce блокирует thread-GPU
    # configure map_location properly
    map_location = {f'cuda:{0}': f'cuda:{rank}'} # map parameters to corresponding process-GPU-device
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


if __name__ == "__main__":

    if(not devices.print_available_devices()):
        exit()

    # https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
    mp.spawn(demo_basic,
             args=(world_size,),
             nprocs=world_size,
             join=True)
