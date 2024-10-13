import os
from flask import Flask, request, jsonify
from summarize import dialogue_processing

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/url", methods=['POST'])
def data_summarization():
    try:
        #get data from DailyDB
        data = request.get_json()
        if not data or 'dialogue' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        dialogue = data['dialogue']
        
        # Summarization
        summarized_dialogue = dialogue_processing(dialogue)
        
        # Return summarized data to superbase
        return jsonify({'summarized_dialogue': summarized_dialogue}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    


"""
Choose a Hosting Platform:

Platforms like AWS, Google Cloud, Heroku, or Azure offer services tailored for deploying Flask applications.
Set Up CI/CD Pipelines:

Automate the deployment process to ensure smooth and consistent releases.
Monitor Performance:

Use monitoring tools (e.g., Prometheus, Grafana) to keep an eye on application performance and uptime.
"""