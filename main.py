import os
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils.dataload import LandClassDataset

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def main():
    device = get_device()
    print(f"Using device: {device}")

    if os.path.isfile('split_file.csv') is False:
        with open('data/land_cover_representation/metadata.csv') as f:
            metadata = pd.read_csv(f)
        metadata = metadata[metadata['split_str'] == 'train']

    transform = T.Compose([
        T.Resize((100, 100)),
        T.ToTensor()
    ])

    dataset = LandClassDataset('data/land_cover_representation', transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
