import torch
from utils.config import Config
from utils.transforms import train_transform
from data.dataset import OlivesDataset
from models.backbone import OliveBackBone
from utils.train_functions import pretrain
import torch.optim as optim

# Load data
train_df = pd.read_csv(Config.train_csv_path)
train_dataset = OlivesDataset(train_df, Config.base_dir, transform=train_transform)

# Set up model, optimizer, loss
device = Config.device
model = OliveBackBone().to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.001)
loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.5)

# Pretrain model
pretrain(model, train_dataset, optimizer, loss_fn, device)
