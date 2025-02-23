from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Back, Style

class ResumeMatcher:
    def __init__(self, sentence_model):
        """Initialize with a sentence transformer model for semantic matching."""
        self.sentence_model = sentence_model
        
        # Score ranges for visual display
        self.SCORE_RANGES = {
            'Excellent': (90, 100, Fore.GREEN),
            'Good':      (70, 89,  Fore.LIGHTGREEN_EX),
            'Fair':      (50, 69,  Fore.YELLOW),
            'Poor':      (30, 49,  Fore.RED),
            'Very Poor': (0,  29,  Fore.LIGHTRED_EX)
        }

    def _create_score_visualization(self, skill: str, score: float) -> str:
        """Create a visual representation of the score."""
        BAR_LENGTH = 50  # Total length of the visualization bar
        filled = int((score / 100) * BAR_LENGTH)
        
        # Determine color based on score
        color = None
        rating = None
        for range_name, (min_val, max_val, range_color) in self.SCORE_RANGES.items():
            if min_val <= score <= max_val:
                color = range_color
                rating = range_name
                break
        
        # Create the bar
        bar = '█' * filled + '░' * (BAR_LENGTH - filled)
        
        # Format the output
        return f"{skill:<15} {color}{bar} {score:>5.1f}% - {rating}{Style.RESET_ALL}"

    def _print_score_legend(self):
        """Print a legend showing score ranges."""
        print("\nScore Ranges:")
        print("=" * 50)
        for range_name, (min_val, max_val, color) in self.SCORE_RANGES.items():
            print(f"{color}{range_name:<10} {min_val:>3}-{max_val:>3}%{Style.RESET_ALL}")
        print("=" * 50)

    def analyze_resume(self, resume_text: str, job_description: str, job_config: Dict) -> Dict:
        """Analyze how well a resume matches required skills using semantic similarity with job context."""
        print("\nStarting skill analysis...")
        
        # Extract required skills from job config
        required_skills = list(job_config['skill_weights'].keys())
        
        print(f"Required skills: {', '.join(required_skills)}")
        print(f"Resume text length: {len(resume_text)} characters")

        # Get semantic scores for all skills with job context
        print("\nCalculating semantic similarity scores...")
        resume_embedding = self.sentence_model.encode([resume_text])[0]
        
        # Create skill contexts by combining each skill with job description
        skill_contexts = [
            f"Skill: {skill}\nJob Context: {job_description}" 
            for skill in required_skills
        ]
        skill_embeddings = self.sentence_model.encode(skill_contexts)
        similarities = cosine_similarity([resume_embedding], skill_embeddings)[0]
        
        # Apply exponential scaling to make scores more extreme
        # This will push high similarities higher and low similarities lower
        def scale_score(similarity: float, weight: float) -> float:
            # Convert similarity from [-1,1] to [0,1]
            normalized = (similarity + 1) / 2
            # Apply exponential scaling
            scaled = np.power(normalized, 2)  # Square the score to make it more extreme
            # Convert to percentage and apply weight
            return float(max(0, min(100, scaled * 100))) * weight

        # Calculate weighted scores with exponential scaling
        skill_scores = {
            skill: scale_score(score, job_config['skill_weights'][skill])
            for skill, score in zip(required_skills, similarities)
        }

        # Sort skills by score
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate overall match
        overall_match = np.mean(list(skill_scores.values()))

        # Print results with visualization
        print("\nSkill Match Analysis")
        print("=" * 80)
        self._print_score_legend()
        print("\nDetailed Skill Scores:")
        print("-" * 80)
        for skill, score in sorted_skills:
            print(self._create_score_visualization(skill, score))
        print("-" * 80)
        print(self._create_score_visualization("OVERALL", overall_match))
        print("=" * 80)

        # Calculate experience and skills matches
        skills_match = np.mean([score for skill, score in skill_scores.items()])
        experience_match = overall_match  # Simplified for now

        # Get missing required skills (those scoring below 50%)
        missing_required = [skill for skill, score in skill_scores.items() if score < 50]

        return {
            'overall_match': overall_match,
            'skills_match': skills_match,
            'experience_match': experience_match,
            'missing_required_skills': missing_required,
            'skill_scores': skill_scores
        }