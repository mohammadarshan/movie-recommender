import sys
import os

# Add project root to Python's search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import RatingsDataset

ds = RatingsDataset("data/ratings.csv")
print("Num examples:", len(ds))
print("Num users:", ds.num_users)
print("Num movies:", ds.num_movies)
print("First example:", ds[0])