import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
import numpy as np


class TorchDataset(InMemoryDataset):
    
    def __init__(self, W, X, transform=None, add_self_loop=True):

        super(TorchDataset, self).__init__('.', transform, None, None)

        adj = torch.from_numpy(np.expand_dims(W, axis=0))
        
        data = Data(edge_index=adj.to(torch.long))
        data.num_nodes = W.shape[0]
        data.adj = adj.to(torch.float)
        
        # Features
        data.x = torch.from_numpy(np.expand_dims(X, axis=0)).to(torch.float)
        self.data, self.slices = self.collate([data])


    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class SequentialSampler():

    def __init__(self, data_source, in_channels, out_channels):
        self.data_source = data_source
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        x = self.data_source[:,:,self.t:self.t+self.in_channels]
        y = self.data_source[:,:,self.t+self.in_channels:self.t+self.in_channels+self.out_channels]
        self.t += 1
        
        if self.t == len(self):
            self.t = 0
            raise StopIteration
        return x, y
    
    def __len__(self):
        return self.data_source.shape[-1]-self.out_channels-self.in_channels

