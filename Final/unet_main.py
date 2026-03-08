# Libraries
import torch.multiprocessing as mp
from global_var import config
from unet_train import train

if __name__ == "__main__":
    world_size = config["ngpu"]
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)