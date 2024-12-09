import openai
from app.test.lexical_feature import total_speech

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