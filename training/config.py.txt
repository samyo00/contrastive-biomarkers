import torch

class Config:
    train_csv_path = '/kaggle/input/the-olives-dataset/OLIVES DATASET/Modified Dataset/TRAIN/Training_Biomarker_Data.csv'
    test_csv_path = '/kaggle/input/the-olives-dataset/test_set_labels.csv'
    base_dir = '/kaggle/input/the-olives-dataset/OLIVES DATASET/Modified Dataset/TRAIN/OLIVES'
    base_dir_test = '/kaggle/input/the-olives-dataset/OLIVES DATASET/Modified Dataset/TEST/RECOVERY/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RANDOM_SEED = 42 
    log = True
