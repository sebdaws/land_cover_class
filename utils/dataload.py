from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np

class LandClassDataset(Dataset):
    def __init__(self, root, split=None, transform=None, metadata_file='metadata_balanced.csv'):
        self.transform = transform

        with open(os.path.join(root, metadata_file)) as f:
            metadata = pd.read_csv(f)
        
        if split != None:
            metadata = metadata[metadata['split_str'] == split]
        self.metadata = metadata
        self.images = []
        self.labels = []
        for _, row in metadata.iterrows():
            img_path = os.path.join(root, row['file_name'])
            if os.path.isfile(img_path):
                self.images.append(img_path)
                self.labels.append(row['y'])
    
    def __len__(self):
        return len(self.images)
    
    def get_num_classes(self):
        return len(set(self.labels))
    
    def get_class_weights(self):
        class_counts = self.metadata['y'].value_counts().sort_index()
        total_samples = len(self.metadata)
        class_weights = total_samples / class_counts
        class_weights = class_weights / class_weights.sum()

        return class_weights
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_raw = np.load(img_name)
        image = Image.fromarray(image_raw[:, :, :3])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
    
        return image, label
    
    
    
