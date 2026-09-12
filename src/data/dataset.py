import pandas as pd
import torch
from torch.utils.data import Dataset

class RatingsDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Build mappings: real userId/movieId -> clean 0-indexed integers
        unique_users = df['userId'].unique()
        unique_movies = df['movieId'].unique()

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}

        self.num_users = len(unique_users)
        self.num_movies = len(unique_movies)

        # Apply the mapping to every row
        self.users = df['userId'].map(self.user_to_idx).values
        self.movies = df['movieId'].map(self.movie_to_idx).values
        self.ratings = df['rating'].values.astype('float32')

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.movies[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32)
        )