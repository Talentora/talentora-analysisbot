import os
import time
import random
from dotenv import load_dotenv
import asyncio
import requests

import numpy as np
from tqdm import tqdm
from colorama import Fore, Style, init
from hume import HumeClient  # Make sure HumeClient is installed/configured
from app.configs.hume_config import HUME_API_KEY, HUME_API_SECRET
from app.controllers.supabase_db import SupabaseDB
from app.controllers.hume_job_manager import JobManager
# For visualizations
import matplotlib.pyplot as plt


#goal : 
# send videos to hume
# download hume data
# process hume data into 3 data frames (generating the aggregates)
# send data files for each model to supabase bucket 
# download data and labels from bucket 
# merge data and labels into 3 separate dataframes 
# train 3 separate SVR models 
# train linear regression late fusion model




def main():
    # Initialize colorama
    init()

    # Initialize env variables
    load_dotenv()

    supabase_client = SupabaseDB()
    hume_client = HumeClient(api_key=HUME_API_KEY)
    hume_job_manager = JobManager(hume_client)
    interview_videos = supabase_client.list_storage_files(
            bucket_id="edited-interviews",
            folder_path="/",
            # limit=2
        )
            
    signed_urls = supabase_client.create_signed_url(bucket_id="edited-interviews", files=interview_videos)

    models_config ={
                "face": {},
                "language": {"granularity": "utterance"},
                "prosody":  {"granularity": "utterance"},
            }

    hume_job_manager.start_job(signed_urls, models_config)

if __name__ == "__main__":
    main()