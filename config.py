import os
import streamlit as st

CHROMA_DB_PATH = "./chroma_db"

def get_secret(key: str) -> str:
    # Try Streamlit secrets first (cloud), fall back to env/.env (local)
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} not found. Add it to Streamlit secrets or your .env file.")
        return value

GROQ_API_KEY = get_secret("GROQ_API_KEY")
