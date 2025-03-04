"""
Configuration settings for job analysis and resume matching.
"""

# Default weights for different categories in overall scoring
DEFAULT_CATEGORY_WEIGHTS = {
    'skills': 0.5,
    'experience': 0.3,
    'education': 0.2
}

# Thresholds for semantic matching
SEMANTIC_MATCH_THRESHOLDS = {
    'required': 0.7,
    'relevant': 0.6,
    'experience': 0.5,
    'education': 0.6,
    'strength': 0.8
}

# Common skill patterns for different job categories
SKILL_PATTERNS = {
    'programming_languages': [
        r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Go|Rust)\b'
    ],
    'frameworks': [
        r'\b(React|Angular|Vue|Django|Flask|Spring|Express|TensorFlow|PyTorch|Scikit-learn)\b'
    ],
    'databases': [
        r'\b(SQL|MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|DynamoDB)\b'
    ],
    'cloud_platforms': [
        r'\b(AWS|Azure|GCP|Google Cloud|Heroku|DigitalOcean)\b'
    ],
    'tools': [
        r'\b(Git|Docker|Kubernetes|Jenkins|CircleCI|Travis|Jira|Confluence)\b'
    ],
    'soft_skills': [
        r'\b(Leadership|Communication|Teamwork|Problem[- ]Solving|Critical[- ]Thinking)\b'
    ],
    'methodologies': [
        r'\b(Agile|Scrum|Kanban|Waterfall|DevOps|CI/CD)\b'
    ]
}

# Degree level mappings
DEGREE_LEVELS = {
    'high school': 1,
    'associate': 2,
    "associate's": 2,
    'bachelor': 3,
    "bachelor's": 3,
    'master': 4,
    "master's": 4,
    'doctorate': 5,
    'phd': 5
}

# Default weights for education scoring
EDUCATION_SCORE_WEIGHTS = {
    'degree_level': 0.4,
    'field': 0.4,
    'gpa': 0.2
}

# Section identification patterns
SECTION_HEADERS = {
    'experience': [
        'experience',
        'work experience',
        'employment history',
        'professional experience',
        'work history'
    ],
    'education': [
        'education',
        'educational background',
        'academic background',
        'academic history',
        'qualifications'
    ],
    'skills': [
        'skills',
        'technical skills',
        'core competencies',
        'expertise',
        'technologies',
        'proficiencies'
    ],
    'projects': [
        'projects',
        'personal projects',
        'professional projects',
        'key projects'
    ],
    'certifications': [
        'certifications',
        'certificates',
        'professional certifications',
        'licenses'
    ]
} 