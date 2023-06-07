import torch
import yaml
from torch_geometric.loader import DataLoader


from data import data_prep
from utils.normalize import normalize_data
from model.simple_GCN import GCNModel
from trainer.train import train


# Read config from YAML file
config_path = '/home/nawaf/spectra_ai/src/config/model_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    
data_list = normalize_data(data_prep.prep_data())


batch_size = config['batch_size']
shuffle = config['shuffle']
in_channels = config['model_params']['in_channels']
hidden_channels1 = config['model_params']['hidden_channels1']
hidden_channels2 = config['model_params']['hidden_channels2']
out_channels = config['model_params']['out_channels']
learning_rate = config['learning_rate']



# Create data loader
loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

# Create model
model = GCNModel(in_channels, hidden_channels1, hidden_channels2, out_channels)

# Create optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Train the model
train(model, loader, optimizer, criterion)

