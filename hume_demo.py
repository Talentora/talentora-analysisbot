import asyncio
import os
import cv2
import base64
import numpy as np
from dotenv import load_dotenv
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.types import StreamFace

load_dotenv()

async def main():
    api_key = os.environ.get("HUME_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set HUME_API_KEY in your environment variables.")
    client = AsyncHumeClient(api_key=api_key)

    model_config = Config(face=StreamFace())

    stream_options = StreamConnectOptions(config=model_config)
    
    async with client.expression_measurement.stream.connect(options=stream_options) as socket:
        cap = cv2.VideoCapture(0)  
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # encode frame to jpeg
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                print("Failed to encode frame.")
                continue

            # convert to base64 string
            image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

            result = await socket.send_file(image_base64)

            face_predictions = result.face.predictions
            # face_mesh = result.facemesh

            if face_predictions:
                prediction = face_predictions[0]
                emotions = prediction.emotions

                emotion_scores = [(emotion.name, emotion.score) for emotion in emotions]
                emotion_scores.sort(key=lambda x: x[1], reverse=True)
                top_three = emotion_scores[:3]

                y0, dy = 30, 30
                for i, (emotion_name, score) in enumerate(top_three):
                    y = y0 + i * dy
                    text = f"{emotion_name}: {score:.2f}"
                    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, (139, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Emotion Detection', frame)
            else:
                print("No face predictions found.")
                cv2.imshow('Emotion Detection', frame)

            # exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
