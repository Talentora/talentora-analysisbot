import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

#fetch
def get_supabase_data(table,select_condition):
    #all inputs are string
    response = supabase.table(table).select(select_condition).execute()
    return response

#insert data
def insert_supabase_data(d):
    try:
        data = supabase.table("summaries").insert(d).execute()
        assert len(data.data) > 0
        return data.data
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

def data_update(d_update):
    #d_update dictionary forat
    data = supabase.table("countries").update(d_update).eq("id", 1).execute()

def data_update_format(table_name,data_for_update,condition):
    #table name: string
    #data for update: dictionary
    response = (
        supabase.table(table_name)
        .update(data_for_update)
        .eq("id", 1) #needs to be change
        .execute()
        )
    return response