import torch
from torch.utils.data import DataLoader
from models.backbone import OliveBackBone
from utils.train_functions import train_epoch, evaluate_epoch, train_and_evaluate
from utils.config import Config
from data.dataset import OlivesDataset
from utils.transforms import train_transform, test_transform
from sklearn.metrics import MultilabelAccuracy

# Load data
train_df = pd.read_csv(Config.train_csv_path)
test_df = pd.read_csv(Config.test_csv_path)
train_dataset = OlivesDataset(train_df, transform=train_transform, unlabelled=False)
test_dataset = OlivesDataset(test_df, transform=test_transform, unlabelled=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

device = Config.device
model = OliveBackBone().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.001)
loss_fn = nn.BCEWithLogitsLoss()

# Load weights
path = '/kaggle/input/cl-weights/cl_weigts.pt'
state_dict = torch.load(path, map_location=Config.device)
model.load_state_dict(state_dict)

# Train and evaluate
train_and_evaluate(model, train_loader, test_loader, loss_fn, optim, MultilabelAccuracy(6).cpu(), device, epochs=100)
