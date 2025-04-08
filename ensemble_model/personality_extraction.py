import os
import numpy as np
from typing import List, Dict
import openai
from dotenv import load_dotenv
from ensemble_model.embedding_generation import generate_embedding
from ensemble_model.data_ingestion import preprocess_text

# Load environment variables from .env file
load_dotenv()

# (Optional) If you plan to use OpenAI's API, configure your API key:
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_personality_traits(transcript: str) -> str:
    """
    Extracts personality traits from the given interview transcript using an LLM.
    
    Parameters:
        transcript (str): The full transcript from the interview.
        
    Returns:
        str: A structured HEXACO personality profile with scores and justifications.
    """
    prompt = (
        "Analyze the following interview transcript and generate a detailed HEXACO personality profile.\n\n"
        "Instructions:\n"
        "1. Focus Exclusively on Observable Behavior: Evaluate the candidate solely based on work-related behaviors and responses in the transcript. Do not incorporate any assumptions based on personal demographics or non-work related factors.\n"
        "2. For Each HEXACO Dimension, Provide:\n"
        "   • A numeric score on a scale from 1 to 5 (where 1 = Very Low, 3 = Moderate, 5 = Very High).\n"
        "   • A concise, evidence-based justification (1–2 sentences) that references specific behaviors in the transcript.\n"
        "   • Do not add any whitespaces inbetween lines of text\n"
        "3. HEXACO Dimensions to Assess:\n"
        "   • Honesty-Humility: Consider the candidate's level of sincerity, fairness, and modesty.\n"
        "   • Emotionality: Evaluate the candidate's sensitivity, anxiety, and emotional reactivity versus their resilience.\n"
        "   • Extraversion: Assess the candidate's sociability, assertiveness, and level of energy.\n"
        "   • Agreeableness: Determine how cooperative, empathetic, and considerate the candidate is.\n"
        "   • Conscientiousness: Judge the candidate's organization, dependability, and diligence.\n"
        "   • Openness to Experience: Examine the candidate's curiosity, creativity, and willingness to embrace new ideas.\n\n"
        "Please output your evaluation as a bullet list with each dimension on a new line. For example:\n"
        "- Honesty-Humility: Score 4 – The candidate consistently provided examples showing fairness and modesty in team settings.\n"
        "- Emotionality: Score 2 – The candidate's responses indicate resilience with minimal displays of anxiety, even in challenging contexts.\n\n"
        "Transcript:\n"
        f"{transcript}\n\n"
        "Candidate HEXACO Profile:"
    )
    
    # Using the new OpenAI API
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional personality psychologist specializing in the HEXACO model. Your task is to evaluate the candidate's personality based solely on observable behaviors in the interview transcript, ignoring any personal identifiers such as age, gender, ethnicity, or socioeconomic background. Focus only on responses that demonstrate work-related behaviors and personality traits."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,  # Increased to accommodate detailed responses
        temperature=0  # deterministic output
    )
    traits_text = response.choices[0].message.content.strip()
    return traits_text

def generate_personality_embedding(transcript: str) -> np.ndarray:
    """
    Generates an embedding for the candidate's personality based on the extracted traits.
    
    Parameters:
        transcript (str): The candidate's interview transcript.
    
    Returns:
        np.ndarray: The embedding vector representing the candidate's personality.
    """
    # Extract personality traits (as a comma-separated string)
    traits_text = extract_personality_traits(transcript)
    # Convert the traits text into an embedding using the embedding generation module.
    embedding = generate_embedding(traits_text)
    return embedding

if __name__ == "__main__":
    sample_transcripts = [
    # Transcript 1: Very Good Fit – Demonstrates consistently high levels of positive traits
    "I consistently seek to collaborate and innovate in everything I do. In my last role, I initiated a team project that improved our workflow efficiency by 30%. "
    "I always ensure that credit is given fairly and actively mediate conflicts to achieve mutually beneficial outcomes. I am proactive in soliciting feedback and readily adapt my approach based on team input. "
    "I also embrace new ideas, constantly participate in brainstorming sessions, and maintain a strong ethical standard in all my interactions.",

    # Transcript 2: Good Fit – Above average but not outstanding across all dimensions
    "I maintain a systematic approach in my work and value clear communication. In my previous position, I developed a reliable method for reviewing tasks and ensuring quality work. "
    "I contribute effectively during team meetings and offer thoughtful suggestions. Although I sometimes prefer working independently, I adapt well when collaboration is required, showing a good balance of creativity and diligence.",

    # Transcript 3: Moderate Fit – Average performance with mixed indications across traits
    "I make an effort to create a positive work environment and have occasionally mediated conflicts among teammates. However, I sometimes struggle under tight deadlines and my responses can be inconsistent. "
    "I demonstrate adequate attention to detail and reliability but occasionally exhibit hesitation when adapting to sudden changes. Overall, my approach is balanced but lacks consistent excellence in any one dimension.",

    # Transcript 4: Poor Fit – Traits may lead to negative consequences in team settings
    "I focus primarily on results and often make decisions quickly without seeking team input. My direct approach sometimes comes off as blunt, and I rarely acknowledge alternative perspectives. "
    "This has occasionally led to conflicts and a breakdown in team collaboration, as I tend to prioritize efficiency over harmonious interaction. My adaptability seems limited when unexpected challenges arise.",

    # Transcript 5: Very Poor Fit – Clear misalignment with desired personality and cultural traits
    "I rely solely on my own judgment and rarely consider others' opinions, even when collaboration is needed. I dismiss feedback and neglect to adjust my methods, which has led to repeated project failures. "
    "My communication is often terse and unempathetic, and I do not show any willingness to adapt or engage with team members. This self-centered approach consistently disrupts team dynamics."
]
    
    for i, transcript in enumerate(sample_transcripts, 1):
        print(f"\nTranscript {i}:")
        traits = extract_personality_traits(transcript)
        print("Extracted Personality Traits:", traits)
        
        personality_emb = generate_personality_embedding(transcript)
        print("Personality Embedding Shape:", personality_emb.shape)