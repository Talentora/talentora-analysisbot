from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import re

class SkillsExtractor:
    """Class for extracting skills from a resume using semantic analysis and NLP."""
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        print(f"Using device: {device}")
        
        try:
            # Initialize zero-shot classifier with specific model and device
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device
            )
        except Exception as e:
            print(f"Warning: Failed to initialize zero-shot classifier on {device}. Falling back to CPU. Error: {str(e)}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device="cpu"
            )

    def extract_and_normalize_skills(self, text: str) -> List[str]:
        """Extract skills using semantic analysis and NLP."""
        if not text:
            return []
            
        # Extract candidate phrases
        doc = self.nlp(text)
        candidates = []
        
        # Get noun phrases and named entities
        for chunk in doc.noun_chunks:
            if self._is_likely_skill(chunk):
                candidates.append(chunk.text)
        
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG', 'GPE']:
                candidates.append(ent.text)

        # Clean and deduplicate candidates
        candidates = list(set([
            c.strip() for c in candidates 
            if len(c.strip()) > 2 and not c.strip().lower() in ['i', 'we', 'they']
        ]))

        if not candidates:
            return []

        try:
            # Classify candidates as skills
            skill_labels = ["programming language", "technology", "tool", "framework", "platform", "skill"]
            results = self.classifier(
                candidates,
                skill_labels,
                multi_label=True
            )

            # Filter for likely skills
            skills = []
            for i, candidate in enumerate(candidates):
                scores = results['scores'][i]
                if max(scores) > 0.7:  # Confidence threshold
                    skills.append(candidate)

        except Exception as e:
            print(f"Warning: Zero-shot classification failed. Using fallback method. Error: {str(e)}")
            # Fallback to pattern matching
            skills = [c for c in candidates if self._is_likely_skill_pattern(c)]

        return self._normalize_skills(skills)

    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize and deduplicate skills."""
        if not skills:
            return []

        # Get embeddings for similarity comparison
        embeddings = self.nlp.pipe(skills)
        vectors = np.array([doc.vector for doc in embeddings])
        
        # Calculate similarity matrix
        similarities = cosine_similarity(vectors)
        
        # Group similar skills
        unique_skills = []
        seen_indices = set()
        
        for i in range(len(skills)):
            if i in seen_indices:
                continue
            
            # Find similar skills
            similar_indices = np.where(similarities[i] > 0.85)[0]
            seen_indices.update(similar_indices)
            
            # Take the shortest skill name from the group
            group_skills = [skills[idx] for idx in similar_indices]
            unique_skills.append(min(group_skills, key=len))
        
        return sorted(list(set(unique_skills)))

    def _is_likely_skill_pattern(self, text: str) -> bool:
        """Check if text matches common skill patterns."""
        from app.configs.job_analysis_config import SKILL_PATTERNS
        
        text = text.lower()
        for category, patterns in SKILL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
        return False

    def _is_likely_skill(self, span) -> bool:
        """Check if a span is likely to be a skill."""
        return (
            not span.text.lower() in ['i', 'we', 'they', 'it'] and
            not span.like_num and
            len(span.text.split()) <= 4 and
            not all(token.is_stop for token in span)
        ) 