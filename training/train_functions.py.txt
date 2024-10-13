import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from utils.transforms import train_transform, test_transform
from models.backbone import OliveBackBone
from utils.config import Config
from data.dataset import OlivesDataset

def delete_all_files_in_folder(folder_path):
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"All files in {folder_path} have been deleted.")
        else:
            print(f"The folder {folder_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def calculate_f1_score(predictions, ground_truth):
    th_preds = (np.array([pred.numpy() for pred in predictions]) > 0.5) * 1
    gts = np.array([gt.numpy() for gt in ground_truth])
    indi_f1 = f1_score(gts.reshape(-1, 6), th_preds.reshape(-1, 6), average=None)
    avg_f1 = indi_f1.mean()
    return avg_f1

# Training and evaluation functions (train_epoch, evaluate_epoch, train_and_evaluate)
