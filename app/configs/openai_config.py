# openai_config.py
import os
from dotenv import load_dotenv
import openai

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def configure_openai(api_key):
    openai.api_key = api_key
