import os
from dotenv import load_dotenv
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.types import StreamFace

load_dotenv()

def hume_model():
    api_key = os.environ.get("HUME_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set HUME_API_KEY in your environment variables.")
    client = AsyncHumeClient(api_key=api_key)

    model_config = Config(face=StreamFace())
    
    stream_options = StreamConnectOptions(config=model_config)


    return client, stream_options
