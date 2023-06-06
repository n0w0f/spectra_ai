import torch

import os
import yaml

# Assume 'config' is the dictionary containing the configuration values
config = {
    'absorption_mean': 0.0,
    'absorption_std': 1.0,
    'positions_mean': [0.0, 0.0, 0.0],
    'positions_std': [1.0, 1.0, 1.0]
}


def normalize_data(data_list):
    # Extract the absorption coefficients and atom positions from the data_list
    absorption_coefficients = [data.y for data in data_list]
    atom_positions = [data.pos for data in data_list]

   # Stack the absorption coefficients and atom positions into tensors
    absorption_tensor = torch.cat(absorption_coefficients, dim=0)
    atom_positions_tensor = torch.cat(atom_positions, dim=0)

    # Calculate the mean and standard deviation of the absorption tensor and atom positions tensor
    absorption_mean = torch.mean(absorption_tensor)
    absorption_std = torch.std(absorption_tensor)
    positions_mean = torch.mean(atom_positions_tensor, dim=0)
    positions_std = torch.std(atom_positions_tensor, dim=0)

    # Update the config file with the normalization values
    config['absorption_mean'] = absorption_mean.item()
    config['absorption_std'] = absorption_std.item()
    config['positions_mean'] = positions_mean.tolist()
    config['positions_std'] = positions_std.tolist()

    # Define the normalization functions
    def normalize_absorption(tensor):
        return (tensor - absorption_mean) / absorption_std

    def normalize_positions(tensor):
        return (tensor - positions_mean) / positions_std

    # Normalize the absorption coefficients and atom positions in each data object
    for data in data_list:
        data.y = normalize_absorption(data.y)
        data.pos = normalize_positions(data.pos)

    # Set the file path for the config.yaml file
    project_dir = os.path.dirname(__file__)
    config_dir = os.path.join(project_dir, 'config')
    config_path = os.path.join(config_dir, 'config.yaml')

    # Create the config directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)

    # Write the config dictionary to the config.yaml file
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    print("Config file created: config.yaml")

    return data_list

