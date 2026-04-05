import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import get_model
from dataset import TuSimpleDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_loss(pred, target):
    smooth = 1.
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

def train():
    dataset = TuSimpleDataset("data.json", "images/")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = get_model().to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for imgs, masks in loader:
            imgs = imgs.float().to(device)
            masks = masks.float().to(device)

            outputs = model(imgs)['out']

            loss = bce(outputs, masks) + dice_loss(torch.sigmoid(outputs), masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

if __name__ == "__main__":
    train()
