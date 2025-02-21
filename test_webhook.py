import json
from flask import Flask
from app.routes.evaluation import handle_webhook

# Create a test Flask app
app = Flask(__name__)

recording_id = "cf6bcc01-14ac-48d5-9473-bbc516522e1c"
job_id = "1d503eab-bb1c-4c46-99c1-17d576724e73"

# Mock the Daily webhook payload for a completed recording
mock_webhook_payload = {
    "type": "batch-processor.job-finished",
    "payload": {
        "id": job_id,  # Replace with a valid job ID from your testing
        "recording_id": recording_id  # Replace with a valid recording ID
    }
}

# Function to test the webhook handling
def test_handle_webhook():
    with app.test_request_context(method='POST', json=mock_webhook_payload):
        response = handle_webhook()
        print(response.get_data(as_text=True))  # Print the response for verification

# Run the test
if __name__ == "__main__":
    test_handle_webhook()