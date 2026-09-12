import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader, random_split
from src.data.dataset import RatingsDataset
from src.models.recommender import RecommenderNet
import mlflow
import json

# ---- Configure MLflow ----
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("movie-recommender")

# ---- Hyperparameters (pulled out as variables so we can log them) ----
EMBEDDING_DIM = 32
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# ---- 1. Load data ----
dataset = RatingsDataset("data/ratings.csv")

# 80/20 train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- 2. Create the model ----
model = RecommenderNet(num_users=dataset.num_users, num_movies=dataset.num_movies, embedding_size=EMBEDDING_DIM)

# ---- 3. Loss function and optimizer ----
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- Start an MLflow run ----
# ---- 4. Training loop ----
with mlflow.start_run() as run:

    # Log hyperparameters once, at the start
    mlflow.log_param("embedding_dim", EMBEDDING_DIM)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("num_epochs", NUM_EPOCHS)
    mlflow.log_param("num_users", dataset.num_users)
    mlflow.log_param("num_movies", dataset.num_movies)

    for epoch in range(NUM_EPOCHS):
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

        # Log metrics after each epoch
        mlflow.log_metric("train_loss", avg_train_loss)
        mlflow.log_metric("val_loss", avg_val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Log metrics for this epoch (step=epoch lets MLflow plot them as a curve)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

    # ---- 5. Save the trained model ----
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/model.pth"
    torch.save(model.state_dict(), model_path)
    print("Model saved to artifacts/model.pth")

    # NEW: Log the saved model file as an MLflow artifact
    mlflow.log_artifact(model_path)

    # Save the ID mappings so the API can translate real IDs -> model indices
    mappings = {
        "user_to_idx": {str(k): v for k, v in dataset.user_to_idx.items()},
        "movie_to_idx": {str(k): v for k, v in dataset.movie_to_idx.items()}
    }
    mappings_path = "artifacts/mappings.json"
    with open(mappings_path, "w") as f:
        json.dump(mappings, f)
    print("Mappings saved to artifacts/mappings.json")

    mlflow.log_artifact(mappings_path)