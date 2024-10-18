import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class OlivesDataset(Dataset):
    
    def __init__(self, df, transform=None, unlabelled=False):
        self.df = df
        self.transform = transform
        self.unlabelled = unlabelled

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.Path
        if self.unlabelled:
            image = Image.open(img_path.replace('/TREX DME', ''))
        else:
            image = Image.open(img_path.replace('//', '/'))
        
        if self.transform:
            tensor_image = self.transform(image)
        if self.unlabelled:
            return tensor_image
        else:
            np_label = row['B1': 'B6'].values.astype(int)
            tensor_label = torch.tensor(np_label, dtype=torch.float32)
            return tensor_image, tensor_label
