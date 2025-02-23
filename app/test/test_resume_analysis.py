import os
from app.services.resume.analyzer import ResumeAnalyzer
from fpdf import FPDF

def create_test_resume(pdf_path: str, content: str):
    """Create a test resume PDF with given content."""
    content = (content
               .replace('•', '-')  # Replace bullets with hyphens
               .replace('"', '"')  # Replace smart quotes
               .replace('"', '"')
               .replace(''', "'")
               .replace(''', "'")
               .replace('–', '-')  # Replace en-dash
               .replace('—', '-')  # Replace em-dash
               )
    
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for line in content.split('\n'):
        line = line.encode('ascii', 'replace').decode('ascii')
        pdf.multi_cell(0, 10, line)
    
    pdf.output(pdf_path)

def create_sample_resumes():
    """Create sample resumes for testing."""
    # Create test directory
    test_dir = "app/test/test_resumes"
    os.makedirs(test_dir, exist_ok=True)
    
    # Sample job description and config for reference
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

    job_config = {
        'skill_weights': {
            'Python': 1.0,
            'JavaScript': 1.0,
            'React': 0.8,
            'AWS': 0.7,
            'CI/CD': 0.6,
            'DevOps': 0.6
        },
        'min_years_experience': 5,
        'required_education': {
            'degree': 'Bachelor',
            'field': 'Computer Science',
            'min_gpa': 3.0
        },
        'category_weights': {
            'skills': 0.5,
            'experience': 0.3,
            'education': 0.2
        }
    }

    # Excellent match resume - Senior Full Stack Developer with exact matches
    excellent_content = """
    JANE SMITH
    Senior Full Stack Developer

    TECHNICAL SKILLS
    - Expert in Python (8 years) and JavaScript/TypeScript (7 years)
    - Advanced React development with Redux and modern frontend practices
    - Extensive AWS experience (EC2, S3, Lambda, CloudFormation)
    - DevOps specialist with CI/CD pipeline expertise
    - Certified AWS Solutions Architect
    - Deep expertise in microservices architecture

    EXPERIENCE
    Senior Full Stack Developer | TechCorp (2020-Present)
    - Lead developer for cloud-native Python microservices
    - Architected and implemented complex React applications
    - Designed and maintained CI/CD pipelines with Jenkins
    - Managed large-scale AWS infrastructure using Terraform
    - Mentored junior developers in Python and JavaScript best practices

    Technical Lead | InnovateHub (2018-2020)
    - Built scalable Python backend services
    - Led migration to AWS cloud infrastructure
    - Implemented automated CI/CD workflows
    - Developed React component libraries
    """
    
    # Poor match resume - Sales Rep with completely different domain
    poor_content = """
    MICHAEL BROWN
    Enterprise Sales Representative

    PROFESSIONAL SUMMARY
    Results-driven sales professional with 10 years of experience in B2B sales.
    Expert in relationship building and exceeding revenue targets.

    SKILLS
    - Enterprise Sales Strategy
    - Client Relationship Management
    - Sales Pipeline Development
    - Contract Negotiation
    - Salesforce CRM Expert
    - Microsoft Office Suite
    - Public Speaking
    - Sales Team Leadership

    EXPERIENCE
    Senior Enterprise Sales Representative | Global Sales Inc. (2019-Present)
    - Exceeded annual sales targets by 150%, generating $5M+ in new revenue
    - Managed portfolio of 50+ enterprise clients
    - Developed and executed strategic sales presentations
    - Led sales training workshops for junior representatives
    - Implemented new Salesforce workflows

    Regional Sales Manager | Sales Pro Corp. (2016-2019)
    - Managed team of 10 sales representatives
    - Developed regional sales strategies
    - Conducted market analysis and competitor research
    - Created sales forecasting models
    """

    # Moderate match resume - Junior dev with some relevant skills but lacking depth
    moderate_content = """
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
    """

    # Create the PDFs
    create_test_resume(os.path.join(test_dir, "excellent_match.pdf"), excellent_content)
    create_test_resume(os.path.join(test_dir, "poor_match.pdf"), poor_content)
    create_test_resume(os.path.join(test_dir, "moderate_match.pdf"), moderate_content)

    # Test the analysis
    analyzer = ResumeAnalyzer()
    for resume_name in ["excellent_match.pdf", "moderate_match.pdf", "poor_match.pdf"]:
        pdf_path = os.path.join(test_dir, resume_name)
        result = analyzer.analyze(pdf_path, job_description, job_config)
        print(f"\nResults for {resume_name}:")
        # print(result)
        if result:
            print(f"Overall match: {result['overall_match']:.2f}")
            print(f"Skills match: {result['skills_match']:.2f}")
            print(f"Experience match: {result['experience_match']:.2f}")
            print(f"Missing skills: {', '.join(result['missing_required_skills'])}")

if __name__ == '__main__':
    create_sample_resumes()