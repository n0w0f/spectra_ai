from torch.utils.data import random_split


def split_data(data_list, train_ratio, val_ratio):
    # Calculate the sizes of train, validation, and test sets
    total_size = len(data_list)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Perform the random split
    train_data, val_data, test_data = random_split(data_list, [train_size, val_size, test_size])

    return train_data, val_data, test_data
