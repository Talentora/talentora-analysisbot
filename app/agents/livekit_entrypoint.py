import os
import json
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from livekit import api
from livekit.agents import cli, WorkerOptions, JobContext
from app.controllers.supabase_db import SupabaseDB

load_dotenv() 

async def entrypoint(ctx: JobContext):
    # 1) start composite egress (MP4) into your Supabase bucket
    egress_req = api.RoomCompositeEgressRequest(
        room_name=ctx.room.name,
        audio_only=False,
        file_outputs=[
            api.EncodedFileOutput(
                file_type=api.EncodedFileType.MP4,
                filepath=f"{ctx.room.name}_{datetime.now(datetime.timezone.utc):%Y%m%d_%H%M%S}.mp4",
                s3=api.S3Upload(
                    bucket=os.getenv("SUPABASE_S3_BUCKET"),
                    region=os.getenv("SUPABASE_S3_REGION"),
                    access_key=os.getenv("SUPABASE_S3_KEY"),
                    secret=os.getenv("SUPABASE_S3_SECRET"),
                ),
            )
        ],
    )
    lk = api.LiveKitAPI()
    await lk.egress.start_room_composite_egress(egress_req)
    await lk.aclose()

    # 2) on shutdown, dump transcript & insert into recordings table
    async def write_and_register():
        ts = datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        transcript = ctx.session.history.to_dict()
        transcript_json = json.dumps(transcript, indent=2)

        supa = SupabaseDB()
        video_path      = egress_req.file_outputs[0].filepath
        transcript_path = f"transcripts/{ctx.room.name}_{ts}.json"

        # upload transcript
        buf = BytesIO(transcript_json.encode())
        supa.upload_file("transcripts", transcript_path, buf)

        # insert row into recordings (fires your Supabase webhook)
        supa.insert_supabase_data("recordings", {
            "room_name":       ctx.room.name,
            "video_path":      video_path,
            "transcript_path": transcript_path,
        })

    ctx.add_shutdown_callback(write_and_register)

    # 3) connect and run agent logic
    await ctx.connect()

if __name__ == "__main__":
    # this will start the worker, listen for room assignments, and call your entrypoint
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
