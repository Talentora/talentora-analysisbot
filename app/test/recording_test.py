import requests
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

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

    def get_access_link(self, recording_id: str) -> str:
        """Get a temporary download link for the recording"""
        url = f"{self.base_url}/recordings/{recording_id}/access-link"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()["download_link"]

    def download_recording(self, recording_id: str, output_dir: str = "downloads") -> str:
        """
        Download the recording to a local file.
        
        Args:
            recording_id: The ID of the recording to download
            output_dir: Directory to save the downloaded file
            
        Returns:
            Path to the downloaded file
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get recording info
        print(f"Getting info for recording {recording_id}...")
        recording_info = self.get_recording_info(recording_id)
        
        # Get download link
        print("Getting download link...")
        download_link = self.get_access_link(recording_id)
        
        # Prepare output filename
        output_path = os.path.join(
            output_dir, 
            f"{recording_info['room_name']}_{recording_id}.mp4"
        )
        
        # Download the file
        print(f"Downloading recording to {output_path}...")
        response = requests.get(download_link, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress tracking
        with open(output_path, 'wb') as f:
            if total_size == 0:
                print("Warning: Content length not provided by server")
                f.write(response.content)
            else:
                downloaded = 0
                total_size_mb = total_size / (1024 * 1024)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = int((downloaded / total_size) * 100)
                        mb_downloaded = downloaded / (1024 * 1024)
                        print(f"\rProgress: {percent}% ({mb_downloaded:.1f}MB / {total_size_mb:.1f}MB)", end="")
        
        print("\nDownload complete!")
        return output_path

def main():
    # Get API key from environment
    api_key = os.environ.get("DAILY_API_KEY")
    if not api_key:
        print("Error: DAILY_API_KEY not found in environment variables")
        return

    # Initialize downloader
    downloader = DailyVideoDownloader(api_key)
    
    # Specify recording ID
    recording_id = "a299ecbb-b660-4fc2-8913-2caa9715f215"
    
    try:
        # Download the recording
        output_path = downloader.download_recording(recording_id)
        print(f"\nRecording saved to: {output_path}")
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error occurred: {e}")
        if e.response.status_code == 404:
            print("Recording not found. Please check the recording ID.")
        elif e.response.status_code == 401:
            print("Authentication failed. Please check your API key.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()