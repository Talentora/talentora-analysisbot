import os
from supabase import create_client

class supabase():

    def __init__(self):
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_KEY")
        self.supabase = create_client(self.url, self.key)
        

    #fetch
    def get_supabase_data(self, table, select_target):
        #all inputs are string
        response = self.supabase.table(table).select(select_target).execute()
        return response

    #insert data
    def insert_supabase_data(self, table, data_for_insert):
        try:
            inserted_data = self.data_insert_format(table, data_for_insert)
            assert len(inserted_data.data) > 0
            return inserted_data.data
        except Exception as e:
            return {"error": str(e)}

    def data_insert_format(self, table, data_for_insert):
        #table: string
        #data for insert: dictionary
        response = (
            self.supabase.table(table)
            .insert(data_for_insert)
            .execute()
        )
        return response

    def update_supabase_data(self, table_name, data_for_update, condition):
        try:
            updated_data = self.data_update_format(table_name,data_for_update,condition)
            assert len(updated_data.data)>0
            return updated_data
        except Exception as e:
            return {"error": str(e)}

    def data_update_format(self, table_name,data_for_update,condition):
        #table name: string
        #data for update: dictionary
        response = (
            self.supabase.table(table_name)
            .update(data_for_update)
            .eq(condition) #"id", 1
            .execute()
            )
        return response