import os
from dotenv import load_dotenv

load_dotenv()

MERGE_ACCOUNT_TOKEN = os.environ.get("MERGE_ACCOUNT_TOKEN")
MERGE_API_KEY = os.environ.get("MERGE_API_KEY")