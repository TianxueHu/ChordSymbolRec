import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data):
        'Initialization'
        pass

    def __len__(self):
        'Get the total length of the dataset'
        pass
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        pass