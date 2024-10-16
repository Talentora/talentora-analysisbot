import asyncio
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.types import StreamFace

load_dotenv()

async def main():
    #
    api_key = os.environ.get("HUME_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set HUME_API_KEY in your environment variables.")
    client = AsyncHumeClient(api_key=api_key)

    model_config = Config(face=StreamFace())

    stream_options = StreamConnectOptions(config=model_config)
    
    async with client.expression_measurement.stream.connect(options=stream_options) as socket:
        image_path = "631a5b3cda08e18ebf63f147_AdobeStock_246344306-scaled.jpeg"
        result = await socket.send_file(image_path)
        face_predictions = result.face.predictions

        if face_predictions:
            prediction = face_predictions[0]
            emotions = prediction.emotions

            emotion_scores = [(emotion.name, emotion.score) for emotion in emotions]

            emotion_scores.sort(key=lambda x: x[1], reverse=True)

            top_three = emotion_scores[:3]

            print("Top three emotions:")
            for emotion_name, score in top_three:
                print(f"{emotion_name}: {score}")

            emotion_names = [item[0] for item in top_three]
            scores = [item[1] for item in top_three]

            img = mpimg.imread(image_path)

            _, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(img)
            axes[0].axis('off')  
            axes[0].set_title('Input Image')

            axes[1].bar(emotion_names, scores, color='skyblue')
            axes[1].set_xlabel('Emotions')
            axes[1].set_ylabel('Scores')
            axes[1].set_title('Top Three Emotions')
            axes[1].set_ylim(0, 1)  

            plt.tight_layout()
            plt.show()
        else:
            print("No face predictions found.")

if __name__ == "__main__":
    asyncio.run(main())