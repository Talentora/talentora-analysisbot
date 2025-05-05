import json
import os
from dotenv import load_dotenv
from summarize import summarize_interview_json

def main():
    """
    Example script to demonstrate the summarize_interview_json function.
    """
    # Load environment variables
    load_dotenv()
    
    # Sample interview JSON (you can also load from a file)
    sample_interview = {
        "room_name": "interview-1746202801193",
        "timestamp": "2025-05-02T16:22:38.782315",
        "conversation": [
            {"Mike": "Hello, I'm Mike from Acme Inc. Thanks for joining this interview today. How are you doing?"},
            {"Lucas Yoon": "Hey, Mike. I'm doing good."},
            {"Mike": "That's great to hear! So, let's get started with the first question. Um, can you tell me about yourself?"},
            {"Lucas Yoon": "Yeah. So my name is Lucas, and I studied computer science at Boston University."},
            {"Mike": "Thanks for sharing that, Lucas. Can you tell me a bit more about your experiences or projects that you've worked on in the field? You know, something that highlights your skills and background?"},
            {"Lucas Yoon": "I'm not comfortable sharing this information. Can you move on to the next question?"},
            {"Mike": "I understand, Lucas. Let's move on to the next question then. So, what is the React framework?"},
            {"Lucas Yoon": "Don't really know about the React framework. I know it's on on JavaScript library that you can people can reuse components of it."},
            {"Mike": "That's a good start, Lucas. React is indeed a JavaScript library that allows developers to build user interfaces with reusable components. Could you elaborate a bit more on how you might approach using React in a project or its benefits?"},
            {"Lucas Yoon": "Um, I don't really have much experience in React. Can you actually tell me a little bit little bit about it?"},
            {"Lucas Yoon": "Yeah. Yeah. So that's the question I had. Yeah. So thank you. Um, my favorite color is blue."},
            {"Lucas Yoon": "Okay."},
            {"Mike": "Alright, Lucas. So, we've gone through the questions I had prepared for this interview. Thank you for your time today. Do you have any questions or anything else you'd like to share before we conclude?"}
        ]
    }
    
    # Process interview transcript
    summary_result = summarize_interview_json(sample_interview)
    
    # Print the result in a pretty format
    print(json.dumps(summary_result, indent=2))
    
    # Save the summary to a file
    with open("interview_summary.json", "w") as f:
        json.dump(summary_result, f, indent=2)
    
    print(f"Summary saved to interview_summary.json")
    
if __name__ == "__main__":
    main() 