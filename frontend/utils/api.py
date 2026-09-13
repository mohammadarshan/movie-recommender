import requests
import pandas as pd
import os

API_BASE_URL = os.getenv("API_BASE_URL","http://localhost:8000")

_movies_df = None

def register(username:str, password: str):
    response = requests.post(
        f"{API_BASE_URL}/register",
        json={"username": username, "password": password}
    )
    return response

def login(username: str, password: str):
    response = requests.post(
        f"{API_BASE_URL}/login",
        json={"username": username, "password": password}
    )
    return response

def get_recommendations(user_id: str, top_n: int = 10):
    response = requests.post(
        f"{API_BASE_URL}/recommend",
        json={"user_id": user_id, "top_n": top_n}
    )
    return response

def get_movie_title(movie_id: str) -> str:
    global _movies_df
    if _movies_df is None:
        _movies_df = pd.read_csv("data/movies.csv", dtype={"movieId": str})

    row = _movies_df[_movies_df["movieId"] == str(movie_id)]
    if not row.empty:
        return row.iloc[0]['title']
    return f'Unknown movie (ID {movie_id})'