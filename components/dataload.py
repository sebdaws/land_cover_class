from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np

class LandClassDataset(Dataset):
    def __init__(self, root, split=None, transform=None, metadata_file='metadata_balanced.csv', use_infrared=False):
        """
        Parameters
        ----------
        root : str
            Root directory of the dataset
        split : str or None
            Split of the dataset to use. If None, all the data is used.
        transform : callable or None
            Image transform to apply to the images
        metadata_file : str
            Name of the metadata file to use
        use_infrared : bool
            Whether to use the infrared image or not
        """
        self.transform = transform
        self.use_infrared = use_infrared

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
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.images)
    
    def get_num_classes(self):
        """
        Returns the total number of unique classes in the dataset.

        Returns:
            int: Number of unique classes
        """
        return len(set(self.labels))
    
    def get_class_weights(self):
        """
        Computes the class weights from the metadata.

        Returns:
            pandas Series: The class weights, indexed by class id.
        """
        class_counts = self.metadata['y'].value_counts().sort_index()
        total_samples = len(self.metadata)
        class_weights = total_samples / class_counts
        class_weights = class_weights / class_weights.sum()
        return class_weights
    
    def get_class_names(self):
        """
        Returns the names of all classes in the dataset, sorted by class ID.

        Returns:
            list: List of class names (strings) ordered by their corresponding class IDs
        """
        class_names = self.metadata[['y', 'land_cover']].drop_duplicates().sort_values('y')
        return list(class_names['land_cover'])
    
    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Parameters:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (image, label) where image is the transformed image data 
                and label is the corresponding class ID
        """
        img_name = self.images[idx]
        image_raw = np.load(img_name)
        
        if self.use_infrared:
            image = Image.fromarray(image_raw[:, :, :4])
        else:
            image = Image.fromarray(image_raw[:, :, :3])
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
    
        return image, label
    