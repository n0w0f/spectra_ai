import pickle
import torch

from utils import create_graph

# Load data from pickle file
with open('/home/nawaf/spectra/mp_data_dicts.pickle', 'rb') as file:
    mp_data_dicts = pickle.load(file)


def prep_data():
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