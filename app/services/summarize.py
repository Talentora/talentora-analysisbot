import openai
from app.test.lexical_feature import total_speech
import json
from typing import Dict, List, Any, Union

def ai_summary(transcript_summary: str, text_eval: dict, job_description: str, emotion_eval_result: dict) -> dict:
    """
    Use ChatGPT to evaluate the candidate as a whole, providing an overall summary and score.
    This incorporates:
    - The transcript summary
    - The text evaluation (including overall score and explanation)
    - The job description
    - The emotion evaluation results (top emotions, overall emotion aggregate score)
    
    It then requests a JSON response following the specified schema.
    """
    
    # Extract top emotions and overall emotion score from the emotion evaluation
    top_emotions = emotion_eval_result.get('overall', {}).get('top_emotions', [])
    overall_emotion_score = emotion_eval_result.get('overall', {}).get('aggregate_score', 0)
    
    # Format top emotions in a human-readable way
    # Example: [{"emotion": "joy", "score": 0.85}, {"emotion": "confusion", "score": 0.62}]
    # We'll just do a simple listing:
    top_emotions_str = ", ".join([f"{e['emotion']} ({e['score']})" for e in top_emotions])
    
    # Construct the user prompt to send to the model
    prompt = (
        "You are evaluating a candidate for a position. Consider the information below:\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Transcript Summary:\n{transcript_summary}\n\n"
        "Text Evaluation:\n"
        f"- Overall Score: {text_eval.get('overall_score', 0)}\n"
        f"- Explanation: {text_eval.get('explanation', '')}\n\n"
        "Emotion Analysis:\n"
        f"- Top Emotions: {top_emotions_str if top_emotions else 'No top emotions detected'}\n"
        f"- Overall Emotion Aggregate Score (0-10): {overall_emotion_score}\n\n"
        "Now, based on all the above information, assess how well the candidate meets the job requirements.\n"
        "Provide a score from 1 to 100 and a brief explanation. The response must follow the given JSON schema."
    )

    response = openai.chat.completions.create(
        temperature=0.0,
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an experienced recruiter assessing a candidate."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "description": "The score from 1 to 100 indicating how well the candidate meets the job requirements.",
                            "type": "integer"
                        },
                        "explanation": {
                            "description": "A detailed explanation of the assessment.",
                            "type": "string"
                        }
                    },
                    "required": ["score", "explanation"],
                    "additionalProperties": False
                }
            }
        }
    )
    
    return response.choices[0].message.content

def summarize_interview_json(interview_json: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize an interview transcript from a JSON input.
    
    Args:
        interview_json: Either a JSON string or a dictionary containing the interview data
                       with keys 'room_name', 'timestamp', and 'conversation'.
    
    Returns:
        A dictionary containing the summary and metadata about the interview.
    """
    # Parse JSON if a string is provided
    if isinstance(interview_json, str):
        try:
            interview_data = json.loads(interview_json)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
    else:
        interview_data = interview_json
    
    # Extract conversation from the interview data
    conversation = interview_data.get("conversation", [])
    if not conversation:
        return {"error": "No conversation found in the interview data"}
    
    # Format the conversation into a readable transcript
    transcript = ""
    speakers = set()
    
    for message in conversation:
        for speaker, text in message.items():
            speakers.add(speaker)
            transcript += f"{speaker}: {text}\n\n"
    
    # Identify the interviewer and candidate
    # Assumption: The first speaker is the interviewer, or a speaker named with common interviewer patterns
    all_speakers = list(speakers)
    interviewer = None
    candidate = None
    
    interviewer_patterns = ["interviewer", "recruiter", "hr", "hiring"]
    
    for speaker in all_speakers:
        lower_speaker = speaker.lower()
        if any(pattern in lower_speaker for pattern in interviewer_patterns):
            interviewer = speaker
            break
    
    # If no interviewer pattern found, assume the first speaker is the interviewer
    if not interviewer and all_speakers:
        interviewer = all_speakers[0]
    
    # Assume the candidate is any speaker that's not the interviewer
    candidates = [s for s in all_speakers if s != interviewer]
    if candidates:
        candidate = candidates[0]  # Just take the first non-interviewer as the candidate
    
    # Prepare the prompt for summarization
    prompt = (
        "You are an assistant summarizing a job interview transcript. Below is the transcript of an interview.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Please provide a concise summary of the candidate's responses, highlighting their:"
        "1. Technical knowledge and skills demonstrated\n"
        "2. Communication abilities\n"
        "3. Experience mentioned\n"
        "4. Any notable strengths or weaknesses\n\n"
        "Format the summary as a JSON object with the following structure:"
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are an expert at summarizing job interviews."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "interview_summary_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "technical_knowledge": {
                                "description": "Summary of technical knowledge demonstrated",
                                "type": "string"
                            },
                            "communication": {
                                "description": "Assessment of communication skills",
                                "type": "string"
                            },
                            "experience": {
                                "description": "Summary of experience mentioned",
                                "type": "string"
                            },
                            "strengths": {
                                "description": "Notable strengths identified",
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "weaknesses": {
                                "description": "Areas for improvement identified",
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "overall_impression": {
                                "description": "Overall impression of the candidate",
                                "type": "string"
                            }
                        },
                        "required": ["technical_knowledge", "communication", "experience", "strengths", "weaknesses", "overall_impression"]
                    }
                }
            }
        )
        
        summary = json.loads(response.choices[0].message.content)
        
        # Add metadata to the result
        result = {
            "summary": summary,
            "metadata": {
                "room_name": interview_data.get("room_name", ""),
                "timestamp": interview_data.get("timestamp", ""),
                "interviewer": interviewer,
                "candidate": candidate,
                "transcript_length": len(transcript),
                "message_count": len(conversation)
            }
        }
        
        return result
    
    except Exception as e:
        return {"error": f"Error generating summary: {str(e)}"}

'''
Will implement in the future
'''
# def dialogue_processing(text_raw, questions):
#     speech_script = total_speech(text_raw)
#     response = summarize_text(speech_script, questions)
#     return response

# def summarize_text(text, questions): #min_qual is a list whose elements are string
#     questions_text = "\n".join([f"Q{i+1}: {question}" for i, question in enumerate(questions)])

#     prompt = ("You are an assistant assessing qualifications of an interviewee based on their answers.\n\n"
#             f"Questions asked during the interview:\n{questions_text}\n\n"
#             f"Transcribed Interview:\n{text}\n\n"
#             "Based on the interview questions, summarize the interview response.")
    
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system", 
#                     "content": "You are a veteran recruiter skilled at assessing interview responses."
#                 },
#                 {
#                     "role": "user", 
#                     "content": prompt
#                 }
#             ]
#         )

#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error in generating summary: {e}"