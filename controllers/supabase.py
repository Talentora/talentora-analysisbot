import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

#fetch
def get_supabase_data():
    response = supabase.table("countries").select("*").execute()
    return response

#insert data
def insert_supabase_data(d):
    try:
        data = supabase.table("summaries").insert(d).execute()
        assert len(data.data) > 0
        return data.data
    except Exception as e:
        return {"error": str(e)}


"""
insert format
response = (
    supabase.table("countries")
    .insert({"id": 1, "name": "Denmark"})
    .execute()
)
"""
def data_update(d_update):
    #d_update dictionary forat
    data = supabase.table("countries").update(d_update).eq("id", 1).execute()

"""
update format
response = (
    supabase.table("countries")
    .update({"name": "Australia"})
    .eq("id", 1)
    .execute()
)
"""
