from typing import Dict, List, Union
import pdfplumber
import spacy
from datetime import datetime
from transformers import pipeline
import torch

from app.models.resume_models import Experience, Education
from app.configs.job_analysis_config import SECTION_HEADERS
from app.services.resume.skills_extractor import SkillsExtractor

class ResumeParser:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.skills_extractor = SkillsExtractor(nlp_model)
        
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        print(f"Using device: {device}")
        
        try:
            # Initialize NER pipeline with specific model and device
            self.ner = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                device=device
            )
        except Exception as e:
            print(f"Warning: Failed to initialize NER on {device}. Falling back to CPU. Error: {str(e)}")
            self.ner = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                device="cpu"
            )

    def parse_pdf(self, pdf_path: str) -> Dict[str, Union[List[Experience], List[Education], List[str]]]:
        """Parse a resume PDF using ML-based approaches."""
        parsed_data = {
            'experiences': [],
            'education': [],
            'skills': [],
            'raw_text': ''
        }
        
        try:
            # Extract text from PDF
            with pdfplumber.open(pdf_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                
                parsed_data['raw_text'] = text
                
                # Split into logical paragraphs
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                # Classify each paragraph into sections
                sections = self._classify_sections(paragraphs)
                
                # Process each section
                if 'education' in sections:
                    parsed_data['education'] = self._extract_education(sections['education'])
                if 'experience' in sections:
                    parsed_data['experiences'] = self._extract_experiences(sections['experience'])
                if 'skills' in sections:
                    parsed_data['skills'] = self.skills_extractor.extract_and_normalize_skills(sections['skills'])
                
                # If no skills were found in skills section, try to extract from entire resume
                if not parsed_data['skills']:
                    parsed_data['skills'] = self.skills_extractor.extract_and_normalize_skills(text)
                
                return parsed_data
        except Exception as e:
            print(f"Error parsing PDF: {str(e)}")
            return parsed_data

    def _classify_sections(self, paragraphs: List[str]) -> Dict[str, str]:
        """Use zero-shot classification to identify resume sections."""
        sections = {
            'education': [],
            'experience': [],
            'skills': []
        }
        
        for paragraph in paragraphs:
            result = self.classifier(
                paragraph,
                candidate_labels=["education", "work experience", "skills"],
                hypothesis_template="This text is about {}."
            )
            
            # Get the most likely section with confidence > 0.6
            if result['scores'][0] > 0.6:
                section_type = result['labels'][0]
                if section_type == "work experience":
                    sections['experience'].append(paragraph)
                else:
                    sections[section_type].append(paragraph)
        
        # Join paragraphs for each section
        return {k: '\n'.join(v) for k, v in sections.items() if v}

    def _extract_experiences(self, text: str) -> List[Experience]:
        """Extract work experiences using NER and semantic analysis."""
        experiences = []
        doc = self.nlp(text)
        
        current_exp = None
        for sent in doc.sents:
            # Look for company names and job titles
            company = None
            title = None
            dates = []
            
            for ent in sent.ents:
                if ent.label_ == 'ORG':
                    company = ent.text
                elif ent.label_ == 'DATE':
                    dates.append(ent.text)
            
            # Look for job titles (usually before company or at start of sentence)
            for chunk in sent.noun_chunks:
                if any(title_word in chunk.text.lower() for title_word in 
                      ['engineer', 'developer', 'manager', 'analyst', 'consultant']):
                    title = chunk.text
                    break
            
            if company and title:
                if current_exp:
                    experiences.append(Experience(**current_exp))
                
                current_exp = {
                    'company': company,
                    'title': title,
                    'start_date': self._parse_date(dates[0]) if dates else datetime.now(),
                    'end_date': self._parse_date(dates[1]) if len(dates) > 1 else None,
                    'description': [],
                    'skills': []
                }
            elif current_exp and sent.text.strip().startswith('-'):
                # This is likely a bullet point describing responsibilities
                current_exp['description'].append(sent.text.strip())
                
                # Extract skills from the description
                skills = self.skills_extractor.extract_and_normalize_skills(sent.text)
                if skills:
                    current_exp['skills'].extend(skills)
        
        if current_exp:
            experiences.append(Experience(**current_exp))
        
        return experiences

    def _extract_education(self, text: str) -> List[Education]:
        """Extract education information using NER and semantic analysis."""
        education_list = []
        doc = self.nlp(text)
        
        current_edu = None
        for sent in doc.sents:
            # Look for degree mentions
            degree_match = None
            for token in sent:
                if any(deg in token.text.lower() for deg in ["bachelor", "master", "phd", "doctorate"]):
                    degree_match = token.text
                    break
            
            if degree_match:
                if current_edu:
                    education_list.append(Education(**current_edu))
                
                # Initialize new education entry
                current_edu = {
                    'institution': '',
                    'degree': degree_match,
                    'field': None,
                    'graduation_date': None,
                    'gpa': None
                }
                
                # Look for institution (usually follows "from" or "at")
                for token in sent:
                    if token.text.lower() in ["from", "at"]:
                        institution = [t.text for t in token.rights if t.ent_type_ == "ORG"]
                        if institution:
                            current_edu['institution'] = institution[0]
                
                # Look for field of study
                for chunk in sent.noun_chunks:
                    if any(field in chunk.text.lower() for field in ["science", "engineering", "business", "arts"]):
                        current_edu['field'] = chunk.text
                        break
            
            # Look for GPA
            if current_edu and "gpa" in sent.text.lower():
                try:
                    gpa = float(sent.text.lower().split("gpa")[1].split("/")[0].strip(": "))
                    current_edu['gpa'] = gpa
                except:
                    pass
            
            # Look for dates
            for ent in sent.ents:
                if ent.label_ == "DATE" and current_edu:
                    current_edu['graduation_date'] = self._parse_date(ent.text)
        
        if current_edu:
            education_list.append(Education(**current_edu))
        
        return education_list

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string into datetime object."""
        date_str = date_str.lower().strip()
        if date_str == 'present':
            return datetime.now()
        try:
            return datetime.strptime(date_str, '%Y')
        except:
            try:
                return datetime.strptime(date_str, '%B %Y')
            except:
                return datetime.now()

    def _identify_sections(self, doc) -> Dict[str, str]:
        """Identify and separate resume sections."""
        sections = {}
        current_section = None
        current_text = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip().lower()
            found_section = None
            for section, headers in SECTION_HEADERS.items():
                if any(header in sent_text for header in headers):
                    found_section = section
                    break
            
            if found_section:
                if current_section and current_text:
                    sections[current_section] = '\n'.join(current_text)
                current_section = found_section
                current_text = []
            elif current_section:
                current_text.append(sent.text.strip())
        
        if current_section and current_text:
            sections[current_section] = '\n'.join(current_text)
        
        return sections

    def _parse_experience_section(self, text: str) -> List[Experience]:
        """Parse the experience section."""
        experiences = []
        doc = self.nlp(text)
        entries = text.split('\n\n')
        
        for entry in entries:
            if not entry.strip():
                continue
                
            entry_doc = self.nlp(entry)
            company = None
            title = None
            for ent in entry_doc.ents:
                if ent.label_ == 'ORG':
                    company = ent.text
                    break
            
            dates = []
            for token in entry_doc:
                if token.like_num and len(token.text) == 4:
                    dates.append(token.text)
            
            skills = self.skills_extractor.extract_skills(entry)
            description = [sent.text.strip() for sent in entry_doc.sents 
                         if sent.text.strip() and not any(date in sent.text for date in dates)]
            
            if company and dates:
                start_date = datetime.strptime(dates[0], '%Y') if dates else None
                end_date = datetime.strptime(dates[1], '%Y') if len(dates) > 1 else None
                
                experiences.append(Experience(
                    company=company,
                    title=title or "Unknown",
                    start_date=start_date,
                    end_date=end_date,
                    description=description,
                    skills=skills
                ))
        
        return experiences

    def _parse_education_section(self, text: str) -> List[Education]:
        """Parse the education section."""
        # [Previous _parse_education_section implementation]
        # Moving implementation details to keep response focused

    def _parse_skills_section(self, text: str) -> List[str]:
        """Parse the skills section."""
        return self.skills_extractor.extract_and_normalize_skills(text) 