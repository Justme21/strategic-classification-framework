from ..datasets import CSVDataset, CSV_DATASET_DICT, TensorDataset
from ..interfaces import BaseDataset

from torch.utils.data import random_split

import math
import warnings

def lookup_filename(filename:str, verbose:bool=False):
    """Allow aliasing for known datasets"""
    if filename in CSV_DATASET_DICT:
        if verbose:
            print(f"{filename} found in lookup: {CSV_DATASET_DICT[filename]}")
        return CSV_DATASET_DICT[filename]
    else:
        return filename

def _split_dataset(dataset:BaseDataset, train:float, val:float, test:float) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    assert math.isclose(train + val + test, 1), f"Error: In dataset args, train ({train}) + val ({val}) + test ({test}) must = 1"

    with warnings.catch_warnings(action="ignore"):
        # If val or test are 0 then this will raise a warning here. But this isn't very interesting, so I'm suppressing it
        train_dset, val_dset, test_dset = random_split(dataset, [train, val, test])

    train_X_Y = train_dset.dataset[train_dset.indices] 
    valid_X_Y = val_dset.dataset[val_dset.indices]
    test_X_Y = test_dset.dataset[test_dset.indices] 

    train_dataset = TensorDataset(*train_X_Y, filename=dataset.filename, standardise=False)
    val_dataset = TensorDataset(*valid_X_Y, filename=dataset.filename, standardise=False)
    test_dataset = TensorDataset(*test_X_Y, filename=dataset.filename, standardise=False)
    return train_dataset, val_dataset, test_dataset

def split_dataset(dataset:BaseDataset, train:float=1.0, val:float=0.0, test:float=0.0, **kwargs) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Split dataset into train, validation, and test datasets"""
    train_dset, val_dset, test_dset = _split_dataset(dataset, train, val, test)

    return train_dset, val_dset, test_dset

def get_dataset(filename:str, target_label:int|str, verbose:bool=False, dataset_args={}) -> BaseDataset|None:
    try:
        if filename.endswith('.csv'):
            dataset = CSVDataset(filename, target_label, verbose=verbose, **dataset_args)
        else:
            print(f"Error: {filename} datatype not supported")
            dataset = None
    except FileNotFoundError:
        print(f"Error: could not find dataset at {filename}")
        dataset = None

    return dataset


