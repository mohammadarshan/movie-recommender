import requests

API_BASE_URL = "http://localhost:8000"

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