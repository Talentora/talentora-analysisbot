from supabase import create_client
from dotenv import load_dotenv
from app.configs.supabase_config import SUPABASE_URL, SUPABASE_KEY
from typing import Optional, List, Dict, Union
from colorama import Fore, Style, init
from tqdm import tqdm
import time

# Initialize colorama
init()

class SupabaseDB:
    def __init__(self):
        print(f"{Fore.CYAN}Initializing Supabase client...{Style.RESET_ALL}")

        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"{Fore.GREEN}✓ Supabase client initialized successfully{Style.RESET_ALL}")

    def get_supabase_data(self, table: str, select_target: str, condition: list = None):
        """
        Fetch data from the given table with optional conditions.
        """
        print(f"{Fore.CYAN}Fetching data from table '{table}'...{Style.RESET_ALL}")
        query = self.client.table(table).select(select_target)
        if condition:
            query = query.eq(condition[0], condition[1])
        response = query.execute()
        print(f"{Fore.GREEN}✓ Data fetched successfully{Style.RESET_ALL}")
        return response

    def insert_supabase_data(self, table: str, data_for_insert: dict) -> dict:
        """
        Insert data into the given table.
        
        Args:
            table (str): The name of the table to insert into
            data_for_insert (dict): The data to insert
            
        Returns:
            dict: Response data or error message
        """
        print(f"{Fore.CYAN}Inserting data into table '{table}'...{Style.RESET_ALL}")
        try:
            with tqdm(total=1, desc="Inserting", unit="record") as pbar:
                response = self.client.table(table).insert(data_for_insert).execute()
                pbar.update(1)
            
            # Check if insert was successful
            if response and response.data:
                print(f"{Fore.GREEN}✓ Data inserted successfully{Style.RESET_ALL}")
                return {"success": True, "data": response.data[0]}
            
            print(f"{Fore.RED}✗ No data inserted{Style.RESET_ALL}")
            return {"success": False, "error": "No data inserted"}
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error inserting data: {str(e)}{Style.RESET_ALL}")
            return {"success": False, "error": str(e)}

    def update_supabase_data(self, table: str, data_for_update: dict, condition: list):
        """
        Update data in the given table with conditions.
        """
        print(f"{Fore.CYAN}Updating data in table '{table}'...{Style.RESET_ALL}")
        try:
            with tqdm(total=1, desc="Updating", unit="record") as pbar:
                response = (
                    self.client.table(table)
                    .update(data_for_update)
                    .eq(condition[0], condition[1])
                    .execute()
                )
                pbar.update(1)
                
            if response.data:
                print(f"{Fore.GREEN}✓ Data updated successfully{Style.RESET_ALL}")
                return response.data
            else:
                print(f"{Fore.RED}✗ No data updated{Style.RESET_ALL}")
                return {"error": "No data updated."}
        except Exception as e:
            print(f"{Fore.RED}✗ Error updating data: {str(e)}{Style.RESET_ALL}")
            return {"error": str(e)}

    def list_buckets(self) -> Optional[List[dict]]:
        """List all storage buckets"""
        try:
            response = self.client.storage.list_buckets()
            if response:
                print(f"{Fore.GREEN}Found {len(response)} buckets{Style.RESET_ALL}")
            return response
        except Exception as e:
            print(f"{Fore.RED}Error listing buckets: {str(e)}{Style.RESET_ALL}")
            return None

    def get_bucket(self, bucket_id: str) -> Optional[dict]:
        """Get details of a specific bucket"""
        try:
            response = self.client.storage.get_bucket(bucket_id)
            if response:
                print(f"{Fore.GREEN}Retrieved bucket: {response.name}{Style.RESET_ALL}")
            return response
        except Exception as e:
            print(f"{Fore.RED}Error retrieving bucket {bucket_id}: {str(e)}{Style.RESET_ALL}")
            return None

    def list_storage_files(self, bucket_id: str, folder_path: str = "", limit: Optional[int] = None) -> Optional[List[str]]:
        """
        List files in a storage bucket with optional folder path and limit.
        
        Args:
            bucket_id (str): The ID of the storage bucket
            folder_path (str): Optional path to a specific folder
            limit (int, optional): Maximum number of files to return
            
        Returns:
            Optional[List[str]]: List of file paths or None if error occurs
        """
        print(f"{Fore.CYAN}Listing files in bucket '{bucket_id}'...{Style.RESET_ALL}")
        try:
            folder_path = folder_path.strip('/')
            options = {"limit": limit} if limit else None
            response = self.client.storage.from_(bucket_id).list(
                path=folder_path,
                options=options
            )
            file_list = [f"{folder_path}/{file['name']}" if folder_path else file['name'] 
                        for file in response if file['id']]
            print(f"{Fore.GREEN}✓ Found {len(file_list)} files{Style.RESET_ALL}")
            return file_list
        except Exception as e:
            print(f"{Fore.RED}✗ Error listing files: {str(e)}{Style.RESET_ALL}")
            return None

    def upload_file(self, bucket_id: str, destination_path: str, file_data) -> dict:
        """
        Upload a file to storage bucket.
        
        Args:
            bucket_id (str): The ID of the storage bucket
            destination_path (str): Path where file should be stored
            file_data: The file data to upload
            
        Returns:
            dict: Success status and response data or error message
        """
        print(f"{Fore.CYAN}Uploading file to '{destination_path}' in bucket '{bucket_id}'...{Style.RESET_ALL}")
        try:
            destination_path = destination_path.strip('/')
            # with tqdm(total=1, desc="Uploading", unit="file") as pbar:
            response = self.client.storage.from_(bucket_id).upload(
                path=destination_path,
                file=file_data,
                file_options={"upsert": "true"}
            )
                # pbar.update(1)
            print(f"{Fore.GREEN}✓ File uploaded successfully{Style.RESET_ALL}")
            # return {"success": True, "data": response}
        except Exception as e:
            print(f"{Fore.RED}✗ Error uploading file{Style.RESET_ALL}")
            # return {"success": False, "error": str(e)}

    def create_signed_url(self, bucket_id: str, files: List[str], expires_in: int = 3600) -> Optional[str]:
        """
        Create a signed URL for file access.
        
        Args:
            bucket_id (str): The ID of the storage bucket
            file_path (str): Path to the file
            expires_in (int): Number of seconds until URL expires
            
        Returns:
            Optional[str]: Signed URL or None if error occurs
        """
        print(f"{Fore.CYAN}Creating signed URL for '{files}'...{Style.RESET_ALL}")
        try:
            response = self.client.storage.from_(bucket_id).create_signed_urls(
                paths=files,
                expires_in=expires_in
            )
            print(f"{Fore.GREEN}✓ Signed URL created successfully{Style.RESET_ALL}")
            urls = [url["signedURL"] for url in response]
            return urls
        except Exception as e:
            print(f"{Fore.RED}✗ Error creating signed URL: {str(e)}{Style.RESET_ALL}")
            return None




if __name__ == "__main__":
    supabase = SupabaseDB()
    buckets = supabase.list_buckets()

    bucket = buckets[0].id

    bucket_details = supabase.get_bucket(bucket)
    print(bucket_details)