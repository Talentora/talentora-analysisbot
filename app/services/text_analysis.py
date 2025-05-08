from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict
import json
import openai
from datetime import datetime
import os
from dotenv import load_dotenv
import app.configs.openai_config as openai_config

openai.api_key = openai_config.OPENAI_API_KEY

# Data Models
class CriteriaScore(BaseModel):
    score: float
    explanation: str
    supporting_quotes: List[str]

class TechnicalAnalysis(BaseModel):
    overall_score: float
    knowledge_depth: CriteriaScore
    problem_solving: CriteriaScore
    best_practices: CriteriaScore
    system_design: CriteriaScore
    testing_approach: CriteriaScore

class CommunicationAnalysis(BaseModel):
    overall_score: float
    clarity: CriteriaScore
    articulation: CriteriaScore
    listening_skills: CriteriaScore
    professionalism: CriteriaScore

class ExperienceAnalysis(BaseModel):
    overall_score: float
    project_complexity: CriteriaScore
    impact: CriteriaScore
    growth: CriteriaScore
    technical_breadth: CriteriaScore

class BehavioralAnalysis(BaseModel):
    overall_score: float
    problem_approach: CriteriaScore
    collaboration: CriteriaScore
    learning_attitude: CriteriaScore
    initiative: CriteriaScore

class AnalysisResult(BaseModel):
    """Complete analysis result in JSON format."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    technical: TechnicalAnalysis
    communication: CommunicationAnalysis
    experience: ExperienceAnalysis
    behavioral: BehavioralAnalysis
    
# Technical analysis prompt
TECHNICAL_PROMPT = """
You are an expert technical interviewer for frontend engineering roles. Analyze the interview transcript to evaluate:
1. Technical knowledge depth (score 0-10)
2. Problem-solving capabilities (score 0-10)
3. Understanding of best practices (score 0-10)
4. System design knowledge (score 0-10)
5. Testing approach (score 0-10)

For each area:
- Assign a score between 0 and 10
- Provide detailed explanation
- Include relevant quotes from the transcript

Calculate overall_score as the average of all scores.
Be objective and thorough in your analysis.

INTERVIEW TRANSCRIPT:
{transcript}

Return a valid JSON object with the following structure (no extra spaces or newlines at the beginning):
{{
  "overall_score": float,
  "knowledge_depth": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "problem_solving": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "best_practices": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "system_design": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "testing_approach": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }}
}}
"""

# Communication analysis prompt
COMMUNICATION_PROMPT = """
You are an expert in evaluating communication skills. Analyze the interview transcript to evaluate:
1. Clarity of explanations (score 0-10)
2. Articulation of complex concepts (score 0-10)
3. Active listening skills (score 0-10)
4. Professional communication (score 0-10)

For each area:
- Assign a score between 0 and 10
- Provide detailed explanation
- Include relevant quotes from the transcript

Calculate overall_score as the average of all scores.
Focus on how effectively the candidate communicates technical concepts.

INTERVIEW TRANSCRIPT:
{transcript}

Return a valid JSON object with the following structure (no extra spaces or newlines at the beginning):
{{
  "overall_score": float,
  "clarity": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "articulation": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "listening_skills": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "professionalism": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }}
}}
"""

# Experience analysis prompt
EXPERIENCE_PROMPT = """
You are an expert in evaluating professional experience. Analyze the interview transcript to evaluate:
1. Project complexity (score 0-10)
2. Impact and results (score 0-10)
3. Career growth (score 0-10)
4. Technical breadth (score 0-10)

For each area:
- Assign a score between 0 and 10
- Provide detailed explanation
- Include relevant quotes from the transcript

Calculate overall_score as the average of all scores.
Focus on the depth and quality of experience, not just duration.

INTERVIEW TRANSCRIPT:
{transcript}

Return a valid JSON object with the following structure (no extra spaces or newlines at the beginning):
{{
  "overall_score": float,
  "project_complexity": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "impact": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "growth": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "technical_breadth": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }}
}}
"""

# Behavioral analysis prompt
BEHAVIORAL_PROMPT = """
You are an expert in evaluating behavioral competencies. Analyze the interview transcript to evaluate:
1. Problem-solving approach (score 0-10)
2. Collaboration skills (score 0-10)
3. Learning attitude (score 0-10)
4. Initiative and ownership (score 0-10)

For each area:
- Assign a score between 0 and 10
- Provide detailed explanation
- Include relevant quotes from the transcript

Calculate overall_score as the average of all scores.
Look for specific examples that demonstrate these competencies.

INTERVIEW TRANSCRIPT:
{transcript}

Return a valid JSON object with the following structure (no extra spaces or newlines at the beginning):
{{
  "overall_score": float,
  "problem_approach": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "collaboration": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "learning_attitude": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }},
  "initiative": {{
    "score": float,
    "explanation": string,
    "supporting_quotes": [string]
  }}
}}
"""

async def analyze_interview_parallel(transcript: str) -> Dict:
    """
    Analyze an interview transcript by running all analysis aspects.
    
    Args:
        transcript (str): The interview transcript to analyze
        
    Returns:
        Dict: Comprehensive analysis of the interview
    """
    try:
        # Analyze technical skills
        print("Sending request to technical analysis...")
        technical_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": TECHNICAL_PROMPT.format(transcript=transcript)}],
            response_format={"type": "json_object"}
        )
        technical_text = technical_response.choices[0].message.content
        print(f"Technical response (first 100 chars): {technical_text[:100]}")
        technical_analysis = json.loads(technical_text.strip())
        print("Technical analysis parsed successfully")
        
        # Analyze communication skills
        print("Sending request to communication analysis...")
        communication_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": COMMUNICATION_PROMPT.format(transcript=transcript)}],
            response_format={"type": "json_object"}
        )
        communication_text = communication_response.choices[0].message.content
        print(f"Communication response (first 100 chars): {communication_text[:100]}")
        communication_analysis = json.loads(communication_text.strip())
        print("Communication analysis parsed successfully")
        
        # Analyze experience
        print("Sending request to experience analysis...")
        experience_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": EXPERIENCE_PROMPT.format(transcript=transcript)}],
            response_format={"type": "json_object"}
        )
        experience_text = experience_response.choices[0].message.content
        print(f"Experience response (first 100 chars): {experience_text[:100]}")
        experience_analysis = json.loads(experience_text.strip())
        print("Experience analysis parsed successfully")
        
        # Analyze behavioral aspects
        print("Sending request to behavioral analysis...")
        behavioral_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": BEHAVIORAL_PROMPT.format(transcript=transcript)}],
            response_format={"type": "json_object"}
        )
        behavioral_text = behavioral_response.choices[0].message.content
        print(f"Behavioral response (first 100 chars): {behavioral_text[:100]}")
        behavioral_analysis = json.loads(behavioral_text.strip())
        print("Behavioral analysis parsed successfully")
        
        # Combine results
        result = {
            "technical": technical_analysis,
            "communication": communication_analysis,
            "experience": experience_analysis,
            "behavioral": behavioral_analysis,
            "metadata": {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "transcript_length": len(transcript)
            }
        }
        
        return result
    except json.JSONDecodeError as je:
        print(f"JSON decode error: {str(je)}")
        raise Exception(f"Error parsing JSON response: {str(je)}")
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception(f"Error during analysis: {str(e)}")

# async def main():
#     from sample_transcripts import BAD_INTERVIEW
 
#     try:
#         # Run parallel analysis
#         analysis = await analyze_interview_parallel(BAD_INTERVIEW["transcript"])
        
#         # Convert to JSON and print
#         result_json = json.dumps(analysis, indent=2)
        
#         # Save to file
#         with open("analysis_result.json", "w") as f:
#             f.write(result_json)
            
#     except Exception as e:
#         print(f"Error during analysis: {str(e)}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
