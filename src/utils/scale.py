import torch
import torchvision.transforms as transforms
import yaml


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

config_file = 'config/config.yaml'
config = load_config(config_file)


def scale_predictions(predictions, config):
    # Get the absorption mean and standard deviation from the config dictionary
    absorption_mean = config.get('absorption_mean', 0.0)
    absorption_std = config.get('absorption_std', 1.0)

    # Define the inverse normalization transformation
    inverse_normalize_absorption = transforms.Normalize(mean=[-absorption_mean / absorption_std], std=[1 / absorption_std])

    # Scale the predictions back to their original values
    scaled_predictions = inverse_normalize_absorption(predictions)

    return scaled_predictions
