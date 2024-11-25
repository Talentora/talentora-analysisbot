import os
import requests

class DailyVideoDownloader:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.base_url = "https://api.daily.co/v1"

    def get_recording_info(self, recording_id: str) -> dict:
        """Get information about a specific recording"""
        url = f"{self.base_url}/recordings/{recording_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_download_link(self, recording_id: str) -> dict:
        """
        Get recording info and download link
        
        Returns:
            Dictionary containing download_link and recording info
        """
        # Get recording info
        recording_info = self.get_recording_info(recording_id)
        
        # Get download link
        url = f"{self.base_url}/recordings/{recording_id}/access-link"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        download_link = response.json()["download_link"]
        print("Sucessfully retrieved download link")
        
        return {
            "download_link": download_link,
            "recording_info": recording_info
        }

# def main():
#     # Get API key from environment
#     api_key = os.environ.get("DAILY_API_KEY")
#     if not api_key:
#         print("Error: DAILY_API_KEY not found in environment variables")
#         return

#     # Initialize downloader
#     downloader = DailyVideoDownloader(api_key)
    
#     # Specify recording ID
#     recording_id = "a299ecbb-b660-4fc2-8913-2caa9715f215"
    
#     try:
#         # Get the download link
#         result = downloader.get_download_link(recording_id)
#         print(f"\nDownload link: {result['download_link']}")
#         print(f"Recording info: {result['recording_info']}")
        
#     except requests.exceptions.HTTPError as e:
#         print(f"HTTP Error occurred: {e}")
#         if e.response.status_code == 404:
#             print("Recording not found. Please check the recording ID.")
#         elif e.response.status_code == 401:
#             print("Authentication failed. Please check your API key.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()