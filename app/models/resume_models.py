from typing import Dict, List, Union, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Experience:
    """Data class for work experience entries."""
    company: str
    title: str
    start_date: datetime
    end_date: Optional[datetime] = None
    description: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)

@dataclass
class Education:
    """Data class for education entries."""
    degree: str
    institution: str
    field: Optional[str] = None
    graduation_date: Optional[datetime] = None
    gpa: Optional[float] = None

@dataclass
class ResumeAnalysisResult:
    """Data class for resume analysis results."""
    skills_match: Dict[str, float]  # Skill -> match score
    experience_match: float  # 0-1 score
    education_match: float  # 0-1 score
    overall_match: float  # 0-1 score
    missing_required_skills: List[str]
    additional_relevant_skills: List[str]
    years_relevant_experience: float
    education_level_match: bool
    suggested_questions: List[str]
    strengths: List[str]
    areas_to_explore: List[str] 