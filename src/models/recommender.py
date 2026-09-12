import torch
import torch.nn as nn

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=32):
        super().__init__()

        # Embedding tables: one row per user/movie, learned during training
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)

        # Small feed-forward network that turns [user_vector + movie_vector] into a single predicted rating
        self.fc = nn.Sequential(
            nn.Linear(embedding_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, movie_ids):
        user_vector = self.user_embedding(user_ids) # shape: (batch, embedding_dim)
        movie_vector = self.movie_embedding(movie_ids) # shape: (batch, embedding_dim)

        combined = torch.cat([user_vector, movie_vector], dim=1) # shape: (batch, embedding_dim*2)

        rating = self.fc(combined) # shape: (batch, 1)
        return rating.squeeze() # shape: (batch,)