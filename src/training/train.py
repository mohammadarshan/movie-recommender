import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader, random_split
from src.data.dataset import RatingsDataset
from src.models.recommender import RecommenderNet

# ---- 1. Load data ----
dataset = RatingsDataset("data/ratings.csv")

# 80/20 train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ---- 2. Create the model ----
model = RecommenderNet(num_users=dataset.num_users, num_movies=dataset.num_movies, embedding_size=32)

# ---- 3. Loss function and optimizer ----
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---- 4. Training loop ----
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for users, movies, ratings in train_loader:
        optimizer.zero_grad()
        predictions = model(users, movies)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for users, movies, ratings in val_loader:
            predictions = model(users, movies)
            loss = criterion(predictions, ratings)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# ---- 5. Save the trained model ----
os.makedirs("artifacts", exist_ok=True)
torch.save(model.state_dict(), "artifacts/model.pth")
print("Model saved to artifacts/model.pth")
