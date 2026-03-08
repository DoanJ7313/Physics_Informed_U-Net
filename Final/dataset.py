# Libraries
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from global_var import config
import numpy as np

class Dataset(Dataset):
    def __init__(self, root, nx=49, ny=21):
        self.root = root
        self.data = sio.loadmat(root)

        # Constants
        self.B = self.data["B"]
        self.D = self.data["D"]
        self.nodes = self.data["nodes"]
        self.shape_func = self.data["shape_func"]
        self.nx = nx
        self.ny = ny

        # Data
        self.t_data_comb = self.data["t_data_comb"]
        self.u_data_comb = self.data["u_data_comb"]

        # Normalize data
        t_max = np.abs(self.t_data_comb).max()
        u_max = np.abs(self.u_data_comb).max()

        self.t_data_comb = self.t_data_comb / t_max
        self.u_data_comb = self.u_data_comb / u_max
        self.nodes = self.nodes / u_max

    def __len__(self):
        return self.t_data_comb.shape[2]            
    
    def __getitem__(self, idx):
        nodes = self.nodes
        t = self.t_data_comb[:, :, idx]
        u = self.u_data_comb[:, :, idx]

        # Reshape to nodal grid
        nodes = nodes.reshape(self.nx, self.ny, 2)
        t = t.reshape(self.nx, self.ny, 2)
        u = u.reshape(self.nx, self.ny, 2)

        t_grid = torch.from_numpy(t).float().permute(2, 0, 1)  
        u_grid = torch.from_numpy(u).float().permute(2, 0, 1)  
        nodes_grid = torch.from_numpy(nodes).float().permute(2, 0, 1)  

        return nodes_grid, t_grid, u_grid
    
    def get_constants(root, nx=49, ny=21):
        data = sio.loadmat(root)
        B = torch.from_numpy(data["B"]).float()
        D = torch.from_numpy(data["D"]).float()
        psi = torch.from_numpy(data["shape_func"]).float()
        psi = psi.view(-1, nx, ny)

        return B, D, psi
    
    # Dataloaders
    def get_dataset(root):
        dataset = Dataset(root=root)
        train_dset, test_dset = train_test_split(dataset, test_size=0.2, random_state=1)

        return train_dset, test_dset
        
    def get_dataloader(dataset, shuffle_data=True):
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle_data, num_workers=config["workers"], drop_last=True)
        
        return dataloader
    
    # Distributed Sampler for multiprocessing
    def get_sampler(dset, world_size, rank):
        sampler = torch.utils.data.distributed.DistributedSampler(dset,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
        return sampler 
