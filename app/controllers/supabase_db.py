import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseDB:
    def __init__(self):
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_KEY")
        self.client: Client = create_client(self.url, self.key)

    def get_supabase_data(self, table: str, select_target: str, condition: list = None):
        """
        Fetch data from the given table with optional conditions.
        """
        query = self.client.table(table).select(select_target)
        if condition:
            query = query.eq(condition[0], condition[1])
        response = query.execute()
        return response

    def insert_supabase_data(self, table: str, data_for_insert: dict):
        """
        Insert data into the given table.
        """
        try:
            response = self.client.table(table).insert(data_for_insert).execute()
            if response.data:
                return response.data
            else:
                return {"error": "No data inserted."}
        except Exception as e:
            return {"error": str(e)}

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