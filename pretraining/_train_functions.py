import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def pretrain(model, train_dataset, optimizer, loss_fn, device, max_iters=100000):
    avg_loss_list = []

    for i, data in tqdm(enumerate(train_dataset)):
        sample1 = train_dataset[0]
        paired_sample = torch.stack([sample1[0], sample1[1]])
        paired_pred = model(paired_sample.to(device))
        inp1 = paired_pred[0].reshape(6, 100)
        inp2 = paired_pred[1].reshape(6, 100)
        label = sample1[2].to(device)

        keep_mask = label != 0
        masked_inp1 = inp1[keep_mask]
        masked_inp2 = inp2[keep_mask]
        masked_label = label[keep_mask]
        loss = loss_fn(masked_inp1, masked_inp2, masked_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss_list.append(loss.item())

        if i % 1000 == 0:
            avg_loss = np.mean(avg_loss_list)
            avg_loss_list = []
            print(avg_loss)

        if i > max_iters:
            break
