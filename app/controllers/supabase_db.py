import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

#fetch
def get_supabase_data(table,select_target):
    #all inputs are string
    response = supabase.table(table).select(select_target).execute()
    return response

#insert data
def insert_supabase_data(table, data_for_insert):
    try:
        inserted_data = data_insert_format(table, data_for_insert)
        assert len(inserted_data.data) > 0
        return inserted_data.data
    except Exception as e:
        return {"error": str(e)}

def data_insert_format(table, data_for_insert):
    #table: string
    #data for insert: dictionary
    response = (
        supabase.table(table)
        .insert(data_for_insert)
        .execute()
    )
    return response

def update_supabase_data(table_name,data_for_update,condition):
    try:
        updated_data = data_update_format(table_name,data_for_update,condition)
        assert len(updated_data.data)>0
        return updated_data
    except Exception as e:
        return {"error": str(e)}

def data_update_format(table_name,data_for_update,condition):
    #table name: string
    #data for update: dictionary
    response = (
        supabase.table(table_name)
        .update(data_for_update)
        .eq(condition) #"id", 1
        .execute()
        )
    return response