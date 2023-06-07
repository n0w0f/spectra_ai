import numpy as np
import torch
from torch_geometric.data import Data

def create_graph(atom,coordinates,absorption_coefficients):
    num_nodes = len(atom)
    
    # Create node features
    x = torch.tensor(atom, dtype=torch.float).view(-1, 1)
    
    # Create edge indices
    edge_index = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edge_index.append([i, j])
            edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    atom = torch.tensor(atom.unsqueeze(1).float())
    pos = torch.tensor(coordinates.float())
    y = torch.tensor([absorption_coefficients]).squeeze().float()


    
    # Create the PyG Data object
    data = Data(atom=atom, pos=pos,y=y, edge_index=edge_index)
    
    return data


def select_data_points(absorption_coefficients, X):
    if X >= len(absorption_coefficients):
        return absorption_coefficients

    # Calculate the interval between selected data points
    interval = len(absorption_coefficients) / X

    # Create an array of indices to select
    indices = np.arange(0, len(absorption_coefficients), interval, dtype=int)

    # Select data points at the specified indices
    selected_data = [absorption_coefficients[i] for i in indices]

    return selected_data