from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Back, Style
from app.configs.job_analysis_config import SECTION_HEADERS
class ResumeMatcher:
    """Class for analyzing a resume against required skills, education, and experience."""
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
        print(job_config)
        """Analyze how well a resume matches required skills, education, and experience."""
        print("\nStarting comprehensive resume analysis...")
        
        # Get skill scores (existing code)
        skill_scores = self._analyze_skills(resume_text, job_description, job_config)
        
        # Analyze education match
        education_scores = self._analyze_education(resume_text, job_config['required_education'])
        
        # Analyze experience match
        experience_scores = self._analyze_experience(resume_text, job_description, job_config)
        
        # Calculate weighted overall match
        weights = job_config['category_weights']
        overall_match = (
            skill_scores['overall_match'] * weights['skills']
            + education_scores['overall_match'] * weights['education']
            + experience_scores['overall_match'] * weights['experience']
        )

        # Print detailed analysis
        self._print_analysis_results(skill_scores, education_scores, experience_scores, overall_match, job_config)

        return {
            'overall_match': overall_match,
            'skills_match': skill_scores['overall_match'],
            'education_match': education_scores['overall_match'],
            'experience_match': experience_scores['overall_match'],
            'skill_scores': skill_scores['skill_scores'],
            'education_details': education_scores,
            'experience_details': experience_scores,
            # 'missing_required_skills': skill_scores['missing_required']
        }

    def _analyze_skills(self, resume_text: str, job_description: str, job_config: Dict) -> Dict:
        """Analyze skills section of resume."""
        print("\nStarting skill analysis...")
        
        # Extract required skills from job config
        required_skills = list(job_config['skill_weights'].keys())
        
        print(f"Required skills: {', '.join(required_skills)}")
        print(f"Job description: {job_description}")
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
        # print("\nSkill Match Analysis")
        # print("=" * 80)
        # self._print_score_legend()
        # print("\nDetailed Skill Scores:")
        # print("-" * 80)
        # for skill, score in sorted_skills:
        #     print(self._create_score_visualization(skill, score))
        # print("-" * 80)
        # print(self._create_score_visualization("OVERALL", overall_match))
        # print("=" * 80)

        # Calculate experience and skills matches
        skills_match = np.mean([score for skill, score in skill_scores.items()])
        experience_match = overall_match  # Simplified for now

        # Get missing required skills (those scoring below 50%)
        missing_required = [skill for skill, score in skill_scores.items() if score < 50]

        return {
            'overall_match': overall_match,
            'skills_match': skills_match,
            'experience_match': experience_match,
            # 'missing_required_skills': missing_required,
            'skill_scores': skill_scores
        }

    def _analyze_education(self, resume_text: str, education_requirements: Dict) -> Dict:
        """Analyze education section of resume."""
        from app.configs.job_analysis_config import DEGREE_LEVELS, SECTION_HEADERS, EDUCATION_SCORE_WEIGHTS
        
        # Extract education section using section headers
        education_section = self._extract_section(resume_text, SECTION_HEADERS['education'])
        if not education_section:
            print("Warning: No education section found in resume")
            return {'overall_match': 0, 'details': 'No education section found'}

        # Normalize the education text
        education_text = education_section.lower()
        required_degree = education_requirements['degree'].lower()
        required_field = education_requirements['field'].lower()

        # Check degree level match with more flexible matching
        degree_level_score = 0
        required_level = DEGREE_LEVELS.get(required_degree, 0)
        
        for degree, level in DEGREE_LEVELS.items():
            # Check for degree name or common variations
            variations = [
                degree,
                degree.replace("'s", ""),
                degree.replace("'s", "s"),
                f"{degree} degree",
                degree[:3]  # Common abbreviations like "BSc", "MSc", etc.
            ]
            
            if any(var in education_text for var in variations):
                if level >= required_level:
                    degree_level_score = 100
                else:
                    # Partial credit for lower degrees
                    degree_level_score = (level / required_level) * 100
                break

        # Enhanced field matching using semantic similarity
        field_variations = [
            required_field,
            f"degree in {required_field}",
            f"major in {required_field}",
            f"{required_field} degree",
            f"{required_field} major"
        ]
        
        field_embeddings = self.sentence_model.encode(field_variations)
        edu_embedding = self.sentence_model.encode([education_text])[0]
        
        # Calculate max similarity across all variations
        field_similarities = cosine_similarity([edu_embedding], field_embeddings)[0]
        field_score = float(max(field_similarities) * 100)

        # Additional points for related fields
        related_fields = {
            'computer science': ['software engineering', 'information technology', 'information systems', 
                               'computer engineering', 'data science', 'artificial intelligence'],
            'engineering': ['mechanical engineering', 'electrical engineering', 'civil engineering',
                          'software engineering', 'computer engineering'],
            'business': ['business administration', 'finance', 'economics', 'management',
                        'marketing', 'accounting'],
            'data science': ['statistics', 'mathematics', 'computer science', 'analytics',
                           'quantitative analysis', 'applied mathematics']
        }
        
        # Check for related fields
        for main_field, related in related_fields.items():
            if required_field in main_field and any(rel in education_text for rel in related):
                field_score = max(field_score, 80)  # Give good score for related fields

        # Calculate weighted score using configured weights
        weights = EDUCATION_SCORE_WEIGHTS
        education_score = (
            degree_level_score * weights['degree_level'] +
            field_score * weights['field']
        ) / (weights['degree_level'] + weights['field'])

        # Normalize final score
        education_score = max(0, min(100, education_score))

        return {
            'overall_match': education_score,
            'degree_level_match': degree_level_score,
            'field_match': field_score,
            'details': {
                'found_degree_level': degree_level_score > 0,
                'degree_score': degree_level_score,
                'field_score': field_score,
                # 'education_text': education_text
            }
        }

    def _analyze_experience(self, resume_text: str, job_description: str, job_config: Dict) -> Dict:
        """Analyze experience section of resume."""
        from app.configs.job_analysis_config import SECTION_HEADERS
        
        # Extract experience section
        experience_section = self._extract_section(resume_text, SECTION_HEADERS['experience'])
        if not experience_section:
            return {'overall_match': 0, 'details': 'No experience section found'}

        # Calculate semantic similarity with job description
        exp_embeddings = self.sentence_model.encode([job_description, experience_section])
        role_similarity = cosine_similarity([exp_embeddings[0]], [exp_embeddings[1]])[0][0]

        # Check years of experience
        years_exp = self._extract_years_experience(experience_section)
        years_match = min(1.0, years_exp / job_config.get('min_years_experience', 1))

        # Check for relevant company/industry experience
        company_relevance = self._analyze_company_relevance(experience_section, job_description)

        # Calculate overall experience score
        experience_score = np.mean([
            role_similarity * 100,  # Role similarity
            years_match * 100,  # Years of experience match
            company_relevance * 100  # Company/industry relevance
        ])

        return {
            'overall_match': experience_score,
            'role_similarity': role_similarity * 100,
            'years_experience': years_exp,
            'years_match': years_match * 100,
            'company_relevance': company_relevance * 100
        }

    def _extract_section(self, text: str, section_headers: List[str]) -> str:
        """Extract a specific section from resume text using flexible matching."""
        text_lower = text.lower()
        
        # Try exact matches first
        for header in section_headers:
            header_lower = header.lower()
            if header_lower in text_lower:
                start_idx = text_lower.find(header_lower)
                
                # Find the next section header
                next_section_idx = float('inf')
                for other_headers in SECTION_HEADERS.values():
                    for other_header in other_headers:
                        if other_header.lower() != header_lower:
                            idx = text_lower.find(other_header.lower(), start_idx + len(header))
                            if idx != -1 and idx < next_section_idx:
                                next_section_idx = idx
                
                # Extract section content
                if next_section_idx == float('inf'):
                    section_text = text[start_idx:]
                else:
                    section_text = text[start_idx:next_section_idx]
                
                # Clean up the extracted text
                lines = section_text.split('\n')
                # Remove the header line and any empty lines
                cleaned_lines = [line.strip() for line in lines[1:] if line.strip()]
                return '\n'.join(cleaned_lines)
        
        # If no exact match found, try fuzzy matching
        # Look for common education-related keywords
        edu_keywords = ['university', 'college', 'institute', 'school', 'bachelor', 'master', 'phd', 
                       'degree', 'major', 'gpa', 'graduated']
        
        lines = text.split('\n')
        edu_section_lines = []
        in_edu_section = False
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if this line contains education keywords
            if any(keyword in line_lower for keyword in edu_keywords):
                in_edu_section = True
                edu_section_lines.append(line)
            elif in_edu_section:
                # If we're in education section, keep adding lines until we hit another section
                if any(header.lower() in line_lower for headers in SECTION_HEADERS.values() for header in headers):
                    break
                if line.strip():  # Only add non-empty lines
                    edu_section_lines.append(line)
        
        return '\n'.join(edu_section_lines)

    def _extract_years_experience(self, experience_text: str) -> float:
        """Extract total years of experience from experience section."""
        import re
        from datetime import datetime
        
        # Look for year patterns (YYYY-YYYY or YYYY-Present)
        year_pattern = r'(\d{4})\s*[-–]\s*(Present|\d{4})'
        matches = re.finditer(year_pattern, experience_text)
        
        total_years = 0
        current_year = datetime.now().year
        
        for match in matches:
            start_year = int(match.group(1))
            end_year = current_year if match.group(2) == 'Present' else int(match.group(2))
            total_years += end_year - start_year
        
        return total_years

    def _analyze_company_relevance(self, experience_text: str, job_description: str) -> float:
        """Analyze relevance of companies/industries in experience."""
        exp_embedding = self.sentence_model.encode([experience_text])[0]
        job_embedding = self.sentence_model.encode([job_description])[0]
        return float(cosine_similarity([exp_embedding], [job_embedding])[0][0])

    def _print_analysis_results(self, skill_scores: Dict, education_scores: Dict, experience_scores: Dict, overall_match: float, job_config: Dict):
        """Print detailed analysis results with category breakdowns."""
        print("\n" + "="*80)
        print("COMPREHENSIVE RESUME ANALYSIS")
        print("="*80)
        
        # Print score legend
        self._print_score_legend()
        
        # Skills Analysis
        print("\nSKILLS ANALYSIS")
        print("-"*80)
        print("Individual Skill Scores:")
        for skill, score in sorted(skill_scores['skill_scores'].items(), key=lambda x: x[1], reverse=True):
            print(self._create_score_visualization(skill, score))
        print("\nSkills Summary:")
        print(f"• Average Skill Score: {skill_scores['skills_match']:.1f}%")
        # print(f"• Missing Skills: {', '.join(skill_scores['missing_required_skills']) or 'None'}")
        print(self._create_score_visualization("SKILLS TOTAL", skill_scores['overall_match']))
        
        # Education Analysis
        print("\nEDUCATION ANALYSIS")
        print("-"*80)
        print("Education Components:")
        print(f"• Degree Level Match: {'✓' if education_scores.get('degree_level_match') else '✗'}")
        print(f"• Field Match: {'✓' if education_scores.get('field_match') else '✗'}")
        print(f"• Content Similarity: {education_scores.get('similarity_score', 0):.1f}%")
        print("\nEducation Scoring:")
        print("• Degree Level: " + ("100%" if education_scores.get('degree_level_match') else "50%"))
        print("• Field Match:  " + ("100%" if education_scores.get('field_match') else "50%"))
        print(f"• Similarity:   {education_scores.get('similarity_score', 0):.1f}%")
        print(self._create_score_visualization("EDUCATION TOTAL", education_scores['overall_match']))
        
        # Experience Analysis
        print("\nEXPERIENCE ANALYSIS")
        print("-"*80)
        print("Experience Components:")
        print(f"• Years of Experience: {experience_scores.get('years_experience', 0):.1f} years")
        print(f"• Required Years: {job_config.get('min_years_experience', 0)} years")
        print(f"• Years Match: {experience_scores.get('years_match', 0):.1f}%")
        print(f"• Role Relevance: {experience_scores.get('role_similarity', 0):.1f}%")
        print(f"• Company/Industry Fit: {experience_scores.get('company_relevance', 0):.1f}%")
        print("\nExperience Scoring:")
        print(self._create_score_visualization("Years Match", experience_scores.get('years_match', 0)))
        print(self._create_score_visualization("Role Match", experience_scores.get('role_similarity', 0)))
        print(self._create_score_visualization("Company Fit", experience_scores.get('company_relevance', 0)))
        print(self._create_score_visualization("EXPERIENCE TOTAL", experience_scores['overall_match']))
        
        # Overall Score Calculation
        print("\nOVERALL SCORE CALCULATION")
        print("-"*80)
        weights = job_config['category_weights']
        print("Category Weights:")
        print(f"• Skills:      {weights['skills']*100:>4.0f}%")
        print(f"• Experience:  {weights['experience']*100:>4.0f}%")
        print(f"• Education:   {weights['education']*100:>4.0f}%")
        print("\nWeighted Scores:")
        skills_weighted = skill_scores['overall_match'] * weights['skills']
        exp_weighted = experience_scores['overall_match'] * weights['experience']
        edu_weighted = education_scores['overall_match'] * weights['education']
        print(f"• Skills:      {skill_scores['overall_match']:>5.1f}% × {weights['skills']:.1f} = {skills_weighted:>5.1f}%")
        print(f"• Experience:  {experience_scores['overall_match']:>5.1f}% × {weights['experience']:.1f} = {exp_weighted:>5.1f}%")
        print(f"• Education:   {education_scores['overall_match']:>5.1f}% × {weights['education']:.1f} = {edu_weighted:>5.1f}%")
        print(f"• Total Score: {skills_weighted:>5.1f}% + {exp_weighted:>5.1f}% + {edu_weighted:>5.1f}% = {overall_match:>5.1f}%")
        
        # Final Overall Score
        print("\nFINAL RESULT")
        print("-"*80)
        print(self._create_score_visualization("OVERALL MATCH", overall_match))
        print("="*80)



if __name__ == "__main__":
    matcher = ResumeMatcher(
        SentenceTransformer('all-MiniLM-L6-v2')
    )
    resume_text = """    
    ALEX WILSON
    Junior Software Developer
    SKILLS
    - Basic Python programming
    - Some JavaScript and React experience
    - HTML, CSS, Bootstrap
    - GitHub basics
    - Learning AWS
    - Familiar with Agile methodology
    EXPERIENCE
    Junior Developer | StartupCo (2023-Present)
    - Assist with Python script maintenance
    - Fix minor bugs in React components
    - Help with code reviews
    - Learning CI/CD concepts
    - Completed AWS fundamental training
    Web Development Intern | WebAgency (2022-2023)
    - Built simple websites using HTML/CSS
    - Learned JavaScript fundamentals
    - Assisted with basic React components
    EDUCATION
    Bachelor of Science in Information Technology | State University (2019-2023)
    - Minor in Computer Science
    - GPA: 3.2
    - Relevant Coursework: Web Development, Database Systems, Programming Fundamentals
    - Senior Project: E-commerce Website with React and Node.js"""
    job_description = """
    Senior Software Engineer Position
    
    We are seeking an experienced Software Engineer with strong Python and JavaScript expertise.
    Requirements:
    - 5+ years of experience in software development
    - Expert-level Python and JavaScript programming
    - Experience with React and modern frontend frameworks
    - Strong background in cloud platforms (AWS/Azure)
    - Experience with CI/CD and DevOps practices
    - Bachelor's degree in Computer Science or related field
    """
    job_config = {'skill_weights': {'Python': 1.0, 'JavaScript': 1.0, 'React': 0.8, 'AWS': 0.7, 'CI/CD': 0.6, 'DevOps': 0.6}, 'min_years_experience': 5, 'required_education': {'degree': 'Bachelor', 'field': 'Computer Science', 'min_gpa': 3.0}, 'category_weights': {'skills': 0.5, 'experience': 0.3, 'education': 0.2}}
    matcher.analyze_resume(resume_text, job_description, job_config)