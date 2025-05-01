# agents/livekit_agent.py
import json
import os
from datetime import datetime
from io import BytesIO

from livekit import api
from livekit.agent import JobContext, LiveKitAgent
from app.controllers.supabase_db import SupabaseDB

async def entrypoint(ctx: JobContext):
    # 1) start recording+video egress into your Supabase bucket
    egress_req = api.RoomCompositeEgressRequest(
        room_name=ctx.room.name,
        audio_only=False,
        file_outputs=[
            api.EncodedFileOutput(
                file_type=api.EncodedFileType.MP4,
                filepath=f"{ctx.room.name}_{datetime.utcnow():%Y%m%d_%H%M%S}.mp4",
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

    # 2) on shutdown: upload transcript + insert DB row
    async def write_and_register():
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        transcript = ctx.session.history.to_dict()
        transcript_json = json.dumps(transcript, indent=2)

        supa = SupabaseDB()
        video_path      = egress_req.file_outputs[0].filepath
        transcript_path = f"transcripts/{ctx.room.name}_{ts}.json"

        # upload transcript
        buf = BytesIO(transcript_json.encode())
        supa.upload_file("transcripts", transcript_path, buf)

        # insert into recordings (fires your Supabase webhook)
        supa.insert_supabase_data("recordings", {
            "room_name":       ctx.room.name,
            "video_path":      video_path,
            "transcript_path": transcript_path,
        })

    ctx.add_shutdown_callback(write_and_register)

    # 3) now connect and run your normal agent logic
    await ctx.connect()
    # … any other bot code …

# ─── livekit-agent bootstrap ────────────────────────────────────────
if __name__ == "__main__":
    LiveKitAgent(entrypoint).run()
