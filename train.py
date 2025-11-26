import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ColorizationDataset
from model import UNet

# -------------------------
# DEVICE SELECTION (GPU/CPU)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# LOAD DATASET
# -------------------------
train_dir = "images/train/train2017"

# limit dataset for testing / full run later
train_dataset = ColorizationDataset(train_dir, limit=None)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

# -------------------------
# MODEL, LOSS, OPTIMIZER
# -------------------------
model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# TRAINING LOOP
# -------------------------
epochs = 10
print("Training started...")

for epoch in range(epochs):
    for L, ab in train_loader:
        L = L.to(device)
        ab = ab.to(device)

        optimizer.zero_grad()
        output = model(L)
        loss = criterion(output, ab)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "colorizer_model.pth")
print("Model saved ✔")
