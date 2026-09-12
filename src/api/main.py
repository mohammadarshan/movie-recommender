import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.models.recommender import RecommenderNet

app = FastAPI(title="Movie Recommender API")

# ---- Load mappings ----
with open("artifacts/mappings.json", "r") as f:
    mappings = json.load(f)

user_to_idx = mappings["user_to_idx"]
movie_to_idx = mappings["movie_to_idx"]

NUM_USERS = len(user_to_idx)
NUM_MOVIES = len(movie_to_idx)

# ---- Load ratings data (to know what each user has already rated) ----
ratings_df = pd.read_csv("data/ratings.csv")

# ---- Load the trained model ----
model = RecommenderNet(num_users=NUM_USERS, num_movies=NUM_MOVIES, embedding_size=32)
model.load_state_dict(torch.load("artifacts/model.pth"))
model.eval() # inference mode, not training

# ---- Request/response schemas ----
class PredictionRequest(BaseModel):
    user_id: str
    movie_id: str

class PredictionResponse(BaseModel):
    user_id: str
    movie_id: str
    predicted_rating: float

class TopNRequest(BaseModel):
    user_id: str
    top_n: int = 5

class MovieRecommendation(BaseModel):
    movie_id: str
    predicted_rating: float

class TopNResponse(BaseModel):
    user_id: str
    recommendations: list[MovieRecommendation]

@app.get("/")
def root():
    return {"message": "Movie Recommender API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    user_key = str(request.user_id)
    movie_key = str(request.movie_id)

    if user_key not in user_to_idx:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {user_key}")
    if movie_key not in movie_to_idx:
        raise HTTPException(status_code=404, detail=f"Unknown movie_id: {movie_key}")

    user_idx = torch.tensor([user_to_idx[user_key]], dtype=torch.long)
    movie_idx = torch.tensor([movie_to_idx[movie_key]], dtype=torch.long)

    with torch.no_grad():
        prediction = model(user_idx, movie_idx)

    return PredictionResponse(
        user_id=user_key,
        movie_id=movie_key,
        predicted_rating=round(prediction.item(), 2)
    )

@app.post("/recommend", response_model=TopNResponse)
def recommend(request: TopNRequest):
    user_key = str(request.user_id)

    if user_key not in user_to_idx:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {user_key}")

    user_idx = user_to_idx[user_key]

    # Movies this user has already rated (skip these in recommendations)
    already_rated = set(
        ratings_df[ratings_df["userId"] == int(user_key)]["movieId"].astype(str)
    )

    # Candidate movies: everything the user HASN'T already rated
    candidate_movie_ids = [
        movie_id for movie_id in movie_to_idx.keys() if movie_id not in already_rated
    ]

    candidate_indices = [movie_to_idx[m] for m in candidate_movie_ids]

    # Run the model on ALL candidates at once (batch prediction, not one-by-one)
    user_tensor = torch.tensor([user_idx] * len(candidate_indices), dtype=torch.long)
    movie_tensor = torch.tensor(candidate_indices, dtype=torch.long)

    with torch.no_grad():
        predictions = model(user_tensor, movie_tensor)

    # Pair each movie with its predicted rating, then sort descending
    scored = list(zip(candidate_movie_ids, predictions.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    top_results = scored[:request.top_n]

    return TopNResponse(
        user_id=user_key,
        recommendations=[
            MovieRecommendation(movie_id=movie_id, predicted_rating=round(min(pred, 5.0), 2))
            for movie_id, pred in top_results
        ]
    )