from typing import Dict, Any
import openai
import json

def extract_q_a_pairs(interview_transcript: str) -> Dict[str, str]:
    """
    Extracts questions and responses from an interview transcript.
    The output is a JSON dictionary where each key is a question
    and its corresponding value is the candidate's response.
    """
    prompt = (
        "Extract all questions and responses from the following interview transcript. "
        "Return a JSON dictionary where each key is a question and its value is the corresponding candidate's response.\n\n"
        f"Interview Transcript:\n{interview_transcript}\n"
    )
    
    response = openai.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "You are an assistant that extracts interview Q&A pairs."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "qa_schema",
                "schema": {
                    "type": "object",
                    "patternProperties": {
                        "^.*$": {
                            "type": "string",
                            "description": "Candidate's response to the corresponding interview question"
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
    )
    
    qa_pairs = json.loads(response.choices[0].message.content)
    return qa_pairs

def evaluate_individual_response(question: str, response_text: str, resume: str, job_description: str) -> Dict[str, Any]:
    """
    Evaluates a candidate's response to a single interview question using their resume and the job description.
    
    Returns a JSON dictionary containing:
      - score (int): Rating (1-100) for how well the response fits the requirements.
      - explanation (str): A detailed explanation that includes sentence-by-sentence analysis,
                           identification of covered topics, strengths, and areas for improvement.
    """
    prompt = (
        "You are an experienced recruiter and educational assessor. Your task is to evaluate a candidate's response "
        "to an interview question by taking into account the candidate's resume and the job description. Provide a detailed "
        "assessment that includes:\n\n"
        "1. **Sentence-by-Sentence Analysis:** For each sentence of the candidate's answer, identify which parts align with "
        "the expected requirements (e.g., experience, skills, and other job-relevant aspects).\n\n"
        "2. **Topic Coverage:** Evaluate if the answer covers the topics implied by the job description and resume. "
        "If certain areas are missing, note these gaps.\n\n"
        "3. **Strengths and Weaknesses:** Highlight the strengths of the response as well as areas that need improvement.\n\n"
        "4. **Overall Score:** Provide an overall score between 1 and 100 that reflects how well the response fits the job requirements.\n\n"
        "Please return your response as a JSON object with the following keys:\n"
        "- 'score': an integer between 1 and 100,\n"
        "- 'explanation': a detailed text explanation of your assessment.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate Resume:\n{resume}\n\n"
        f"Interview Question:\n{question}\n\n"
        f"Candidate's Response:\n{response_text}\n\n"
        "Provide your analysis now."
    )

    evaluation = openai.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "You are an experienced recruiter evaluating candidate responses with detailed analysis."},
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
                            "description": "A detailed explanation of the assessment, including sentence-by-sentence analysis, "
                                           "identified strengths, weaknesses, and any missing key topics.",
                            "type": "string"
                        }
                    },
                    "required": ["score", "explanation"],
                    "additionalProperties": False
                }
            }
        }
    )

    return json.loads(evaluation.choices[0].message.content)


def evaluate_candidate(interview_transcript: str, resume: str, job_description: str) -> Dict[str, Any]:
    """
    Overall evaluation workflow:
      1. Extracts all Q&A pairs from the interview transcript.
      2. Evaluates each candidate's response based on their resume and job description.

    Args:
        interview_transcript (str): Full interview transcript.
        resume (str): Candidate's resume.
        job_description (str): Job description for evaluation.

    Returns:
        Dict[str, Any]: Dictionary with questions as keys and evaluation results (score and explanation) as values.
    """
    # Phase 1: Extract Q&A pairs
    qa_pairs = extract_q_a_pairs(interview_transcript)
    
    # Phase 2: Evaluate each response individually
    evaluations = {}
    for question, response_text in qa_pairs.items():
        evaluation_result = evaluate_individual_response(question, response_text, resume, job_description)
        evaluations[question] = evaluation_result
    
    return evaluations

# Example usage:
if __name__ == "__main__":
    interview_transcript = "..."  # your interview transcript text here
    resume = "..."                # candidate's resume text here
    job_description = "..."       # job description text here

    candidate_evaluations = evaluate_candidate(interview_transcript, resume, job_description)
    print(json.dumps(candidate_evaluations, indent=2))
