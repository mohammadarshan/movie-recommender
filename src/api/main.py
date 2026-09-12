import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.recommender import RecommenderNet

app = FastAPI(title="Movie Recommender API")

# ---- Load mappings ----
with open("artifacts/mappings.json", "r") as f:
    mappings = json.load(f)

user_to_idx = mappings["user_to_idx"]
movie_to_idx = mappings["movie_to_idx"]

NUM_USERS = len(user_to_idx)
NUM_MOVIES = len(movie_to_idx)

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