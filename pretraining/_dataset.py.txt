import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class OlivesDataset(Dataset):
    def __init__(self, df, base_path, transform=None):
        self.df = df
        self.transform = transform
        self.df_path = base_path

    def __len__(self):
        return len(self.df) * 1e20

    def __getitem__(self, idx):
        sampled_pair = self.df.sample(2)
        sample_pair = sampled_pair[['B1', 'B2', 'B3', 'B4', 'B5', 'B6']]
        milse = (sample_pair.values.sum(axis=0))
        milse[milse == 1] = -1
        milse[milse == 2] = 1

        img1_path = self.df_path + list(sampled_pair['Path'])[0]
        img2_path = self.df_path + list(sampled_pair['Path'])[1]

        image1 = Image.open(img1_path)
        image2 = Image.open(img2_path)

        if self.transform:
            tensor_image1 = self.transform(image1)
            tensor_image2 = self.transform(image2)

        tensor_label = torch.tensor(milse, dtype=torch.float32)
        
        return tensor_image1, tensor_image2, tensor_label
