import streamlit as st
from utils.api import register, login

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

# ---- Session state: tracks whether the user is logged in, across page navigation ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None

st.title("🎬 Movie Recommender")

if st.session_state.logged_in:
    st.success(f"Logged in as {st.session_state.username}")
    st.write("Use the sidebar to navigate to your recommendations.")

    if st.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.rerun()

else:
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")

            if submitted:
                response = login(username, password)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.logged_in = True
                    st.session_state.username = data["username"]
                    st.session_state.user_id = data["id"]
                    st.rerun()
                else:
                    st.error(response.json().get("detail", "Login failed. Please try again."))

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username", key="reg_username")
            new_password = st.text_input("Choose a password", type="password", key="reg_password")
            submitted = st.form_submit_button("Register")

            if submitted:
                response = register(new_username, new_password)
                if response.status_code == 200:
                    st.success("Registration successful! You can now log in.")
                else:
                    st.error(response.json().get("detail", "Registration failed. Please try again."))


    