import pandas as pd

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

print("Ratings shape:", ratings.shape)
print(ratings.head())
print()
print("Movies shape:", movies.shape)
print(movies.head())
print()
print("Number of unique users:", ratings["userId"].nunique())
print("Number of unique movies:", ratings["movieId"].nunique())