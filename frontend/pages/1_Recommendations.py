import streamlit as st
from utils.api import get_recommendations, get_movie_title

st.set_page_config(page_title="Recommendations", page_icon="🍿")

st.title("🍿 Your Recommendations")

# ---- Guard: redirect-like behavior if not logged in ----
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please log in first from the Home page.")
    st.stop()

st.write(f"Showing recommendations for **{st.session_state.username}**")

top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations"):
    with st.spinner("Scoring movies..."):
        response = get_recommendations(
            user_id=str(st.session_state.movielens_user_id),
            top_n=top_n
        )

    if response.status_code == 200:
        data = response.json()
        recommendations = data["recommendations"]

        st.success(f"Found {len(recommendations)} recommendations!")

        for i, rec in enumerate(recommendations, start=1):
            title = get_movie_title(rec['movie_id'])
            st.write(f"**{i}. {title}** — Predicted rating: ⭐ {rec['predicted_rating']}")
    else:
        st.error("Failed to fetch recommendations. Is the API running?")