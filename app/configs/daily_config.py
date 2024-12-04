import os
from dotenv import load_dotenv

load_dotenv()

DAILY_API_KEY=os.environ.get("DAILY_API_KEY")
DAILY_WEBHOOK_SECRET = os.environ.get("DAILY_WEBHOOK_SECRET")