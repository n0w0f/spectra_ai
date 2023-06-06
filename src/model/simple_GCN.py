import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.conv3 = GCNConv(hidden_channels2, out_channels)
        self.regressor = nn.Linear(out_channels, 2001)

    def forward(self, x,batch, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        h = global_mean_pool(x, batch)
        x=self.regressor(h) 
        return x