"""
This module provides functionality for analyzing and evaluating candidate responses
against job descriptions using AI-powered assessment.
"""

from typing import List
import openai
import json

def total_speech(text_raw: List[str]) -> str:
    """
    Combines multiple text segments into a single continuous speech script.

    Args:
        text_raw (List[str]): List of text segments/responses to combine

    Returns:
        str: Combined text segments as a single string with spaces between segments
    """
    speech_script = " ".join(text_raw)
    return speech_script

def text_evaluation(text_raw: List[str], job_description: str) -> str:
    """
    Evaluates a candidate's responses against a job description.

    Args:
        text_raw (List[str]): List of candidate's responses/text segments
        job_description (str): The job description to evaluate responses against

    Returns:
        str: JSON string containing evaluation results with score and explanation
    """
    # Combine all responses into one text
    speech_script = total_speech(text_raw)
    # Get AI evaluation of the responses
    result = evaluate_responses(speech_script, job_description)
    return result

def evaluate_responses(text: str, job_description: str) -> str:
    """
    Uses OpenAI's GPT model to evaluate how well candidate responses match a job description.

    Args:
        text (str): The candidate's combined responses
        job_description (str): The job description to evaluate against

    Returns:
        str: JSON string containing:
            - score (int): 1-100 rating of how well candidate meets requirements
            - explanation (str): Brief explanation of the assessment

    Note:
        The response format is enforced through a JSON schema to ensure consistent output.
    """
    # Construct the prompt for the AI model
    prompt = (
        "You are a recruiter assessing a candidate's responses based on the job description.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate's Responses:\n{text}\n\n"
        "Evaluate how well the candidate's responses align with the job description."
        " You must provide a score from 1 to 100 based on how well they meet the job requirements."
        " You must also provide a brief explanation of your assessment after the score."
    )

    # Define the expected response structure using JSON schema
    response = openai.chat.completions.create(
        temperature=0.0,
        model="gpt-4o-mini",  # Using GPT-4 mini model for evaluation
        messages=[
            # Set the AI's role as a recruiter
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

    # Extract and return the evaluation result
    result = response.choices[0].message.content
    return result
