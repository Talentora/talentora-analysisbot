import asyncio
import cv2
import base64
import numpy as np
from configs.hume_config import hume_model


async def main():
    client, stream_options = hume_model()
    
    async with client.expression_measurement.stream.connect(options=stream_options) as socket:
        cap = cv2.VideoCapture(0)  
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")

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
