from supabase import create_client
from dotenv import load_dotenv
from app.configs.supabase_config import SUPABASE_URL, SUPABASE_KEY

class SupabaseDB:
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def get_supabase_data(self, table: str, select_target: str, condition: list = None):
        """
        Fetch data from the given table with optional conditions.
        """
        query = self.client.table(table).select(select_target)
        if condition:
            query = query.eq(condition[0], condition[1])
        response = query.execute()
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
        try:
            response = self.client.table(table).insert(data_for_insert).execute()
            
            # Check if insert was successful
            if response and response.data:
                return {"success": True, "data": response.data[0]}
            
            return {"success": False, "error": "No data inserted"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_supabase_data(self, table: str, data_for_update: dict, condition: list):
        """
        Update data in the given table with conditions.
        """
        try:
            response = (
                self.client.table(table)
                .update(data_for_update)
                .eq(condition[0], condition[1])
                .execute()
            )
            if response.data:
                return response.data
            else:
                return {"error": "No data updated."}
        except Exception as e:
            return {"error": str(e)}