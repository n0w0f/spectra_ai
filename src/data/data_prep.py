import pickle
import torch
import yaml


from utils import create_graph
from utils.normalize import normalize_data
from utils.split_data import split_data

# Load data from pickle file
with open('/home/nawaf/spectra/mp_data_dicts.pickle', 'rb') as file:
    mp_data_dicts = pickle.load(file)


config_path = '/home/nawaf/spectra_ai/src/config/model_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)



def data_to_graph():
    # Prepare data for Torch Geometric
    data_list = []
    for mp_data_dict in mp_data_dicts:
        atomic_numbers = torch.tensor(mp_data_dict['structure'].atomic_numbers)
        coordinates = torch.tensor(mp_data_dict['structure'].cart_coords)
        energies = mp_data_dict['energies']
        absorption_coefficients = mp_data_dict['absorption_coefficient']

        # Check if atomic_numbers and coordinates have the same length
        if len(atomic_numbers) != len(coordinates):
            print(f"Size mismatch for atomic_numbers and coordinates in MPID {mp_data_dict['mpid']}. Skipping this data point.")
            continue
        
        data = create_graph.create_graph(atomic_numbers,coordinates,absorption_coefficients)
        data_list.append(data)
    return data_list

def prep_data():
    train_ratio = config['split_ratio']['train']
    val_ratio = config['split_ratio']['val']
    to_be_normalized_data_list = data_to_graph()
    normalized_dataset = normalize_data(to_be_normalized_data_list)
    train_data, val_data, test_data = split_data(normalized_dataset,train_ratio,val_ratio)
    return train_data, val_data, test_data

