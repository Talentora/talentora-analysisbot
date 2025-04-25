import PyPDF2
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from bs4 import BeautifulSoup
from app.services.resume.matcher import ResumeMatcher

class ResumeAnalyzer:
    """Class for extracting text from a resume and analyzing it against required skills (skill analysis done with Resume Matcher)."""
    def __init__(self):
        """Initialize the resume analyzer with required models."""
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.matcher = ResumeMatcher(self.sentence_model)
    
    def extract_text_from_html(self, html_string):
        # Parse the HTML
        soup = BeautifulSoup(html_string, 'html.parser')
        
        # Get all text, stripping whitespace and joining with newlines
        text = '\n'.join(line.strip() for line in soup.stripped_strings)
        
        return text

    def analyze(self, resume_path: str, job_description: str, job_config: Dict) -> Dict:
        """
        Analyze a resume against required skills.
        
        Args:
            resume_path: Path to the resume PDF
            required_skills: List of required skills to match against
            
        Returns:
            Dict containing match results
        """
        try:
            print(f"\nAnalyzing resume file: {resume_path}")
            
            # Extract text from PDF
            print("Extracting text from resume...")
            resume_text = self._extract_text(resume_path)
            print(f"Text extracted from resume: {resume_path}")
 
            if not resume_text.strip():
                print(f"Warning: No text extracted from {resume_path}")
                return None
            
            # Analyze the resume text against required skills
            cleaned_job_description = self.extract_text_from_html(job_description) #clean the HTML tags from job description 
            return self.matcher.analyze_resume(resume_text, cleaned_job_description, job_config)
            
        except FileNotFoundError:
            print(f"Error: Resume file not found at {resume_path}")
            return None
        except Exception as e:
            print(f"Error analyzing resume {resume_path}: {str(e)}")
            return None

    def _extract_text(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text 