from app.test.lexical_feature import text_evaluation

from app.services.response_analysis import text_evaluation
import json

def response_eval(text_raw, job_description):
    text_result_json = text_evaluation(text_raw, job_description)
    text_result = json.loads(text_result_json)

    overall_score = text_result.get('score', 0)
    explanation = text_result.get('explanation', '')

    result_summary = {
        'overall_score': overall_score,
        'explanation': explanation
    }

    return result_summary

# will implement at later iterations
def lexical_eval(text_raw, questions, min_qual, preferred_qual):
    text_result = text_evaluation(text_raw, questions, min_qual, preferred_qual)
    min_qual_scores = text_result['minimum_qualification'].values()
    pref_qual_scores = text_result['preferred_qualification'].values()
    
    min_avg_score = sum(min_qual_scores) / len(min_qual_scores) if min_qual_scores else 0
    pref_avg_score = sum(pref_qual_scores) / len(pref_qual_scores) if pref_qual_scores else 0
    
    # Give minimum qualifications more weight (60%) than preferred qualifications (40%) -- adjust weights by recruiter
    overall_score = (min_avg_score * 0.6) + (pref_avg_score * 0.4)
    overall_score = round(overall_score, 2)

    result_summary = {
        'overall_score': overall_score,
        "min_qual": text_result['minimum_qualification'],
        "pref_qual": text_result['preferred_qualification'],
    }

    return result_summary

"""
def categorize_score(score):
        if score > 80:
            return "great"
        elif score > 60:
            return "good"
        elif score > 40:
            return "ok"
        elif score > 20:
            return "bad"
        else:
            return "poor"
"""
