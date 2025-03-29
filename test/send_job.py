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




# def main():
#     # Initialize colorama
#     init()

#     # Initialize env variables
#     load_dotenv()

#     supabase_client = SupabaseDB()
#     hume_client = HumeClient(api_key=HUME_API_KEY)
#     hume_job_manager = JobManager(hume_client)
#     interview_videos = supabase_client.list_storage_files(
#             bucket_id="edited-interviews"
#         )
            
#     signed_urls = supabase_client.create_signed_url(bucket_id="edited-interviews", files=interview_videos)
#     print(f"created {len(signed_urls)} unique urls...")
 
#     models_config ={
#                 "face": {},
#                 "language": {"granularity": "utterance"},
#                 "prosody":  {"granularity": "utterance"},
#             }

#     hume_job_manager.start_job(signed_urls, models_config)
def main():
    # Initialize colorama
    init()

    # Initialize env variables
    load_dotenv()

    supabase_client = SupabaseDB()
    hume_client = HumeClient(api_key=HUME_API_KEY)
    hume_job_manager = JobManager(hume_client)
    
    # List all files in the "edited-interviews" bucket
    interview_videos = supabase_client.list_storage_files(bucket_id="edited-interviews")
    
    # Define the target file names
    target_files = {"PP11.mp4", "PP12.mp4", "PP13.mp4", "PP14.mp4", "PP15.mp4", "PP16.mp4", "PP17.mp4"}
    # Filter the list to only include the target files
    filtered_videos = [file for file in interview_videos if file in target_files]
            
    signed_urls = supabase_client.create_signed_url(bucket_id="edited-interviews", files=filtered_videos)
    print(f"Created {len(signed_urls)} unique URLs...")
 
    models_config = {
        "face": {},
        "language": {"granularity": "utterance"},
        "prosody":  {"granularity": "utterance"},
    }

    hume_job_manager.start_job(signed_urls, models_config)
if __name__ == "__main__":
    main()