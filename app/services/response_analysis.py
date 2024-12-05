from typing import List
import openai
import json

def total_speech(text_raw: List[str]):
    speech_script = " ".join(text_raw)
    return speech_script

def text_evaluation(text_raw: List[str], job_description: str):
    speech_script = total_speech(text_raw)
    result = evaluate_responses(speech_script, job_description)
    return result

def evaluate_responses(text: str, job_description: str):
    prompt = (
        "You are a recruiter assessing a candidate's responses based on the job description.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate's Responses:\n{text}\n\n"
        "Evaluate how well the candidate's responses align with the job description."
        " You must provide a score from 1 to 100 based on how well they meet the job requirements."
        " You must also provide a brief explanation of your assessment after the score."
    )

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
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
                            "description": "A brief explanation of the assessment.",
                            "type": "string"
                        }
                    },
                    "required": ["score", "explanation"],
                    "additionalProperties": False
                }
            }
        }
    )

    result = response.choices[0].message.content
    return result
