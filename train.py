import torch, os
import matplotlib.pyplot as plt
from dataset import ColorizationDataset
from model import UNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_dir = "images/train/train2017"
val_dir = "images/val/val2017"

train_dataset = ColorizationDataset(train_dir, limit=10000)
val_dataset = ColorizationDataset(val_dir, limit=2000)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

model = UNet().to(device)

# -------- PARAMETER COUNT --------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
train_losses = []
val_losses = []

print("Training started...")

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for L, ab in train_loader:
        L, ab = L.to(device), ab.to(device)

        optimizer.zero_grad()
        output = model(L)
        loss = criterion(output, ab)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ----- VALIDATION -----
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for L, ab in val_loader:
            L, ab = L.to(device), ab.to(device)
            output = model(L)
            total_val_loss += criterion(output, ab).item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "colorizer_model.pth")

print("Training completed and Model saved.")

# -------- PLOT GRAPH --------
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("loss_graph.png")
plt.show()