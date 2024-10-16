import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

#fetch
response = supabase.table("countries").select("*").execute()

#insert format
response = (
    supabase.table("countries")
    .insert({"id": 1, "name": "Denmark"})
    .execute()
)

#update
response = (
    supabase.table("countries")
    .update({"name": "Australia"})
    .eq("id", 1)
    .execute()
)
