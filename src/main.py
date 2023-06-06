import torch
from torch_geometric.loader import DataLoader


from data import data_prep
from utils.normalize import normalize_data
from model.simple_GCN import GCNModel
from trainer.train import train

data_list = normalize_data(data_prep.prep_data())


loader = DataLoader(data_list, batch_size=1, shuffle=True)
model = GCNModel(in_channels=4, hidden_channels1=32, hidden_channels2=64, out_channels=128)  # Adjust the input and output channels accordingly
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()

train(model,loader,optimizer,criterion)

