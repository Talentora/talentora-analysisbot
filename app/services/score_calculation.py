from app.services.lexical_feature import text_evaluation

def eval_result(text_raw, questions, min_qual, preferred_qual):
    #input: 
    #output: dictionary of scores (total, individual scores from different eval method)
    total = 0
    text_result = text_evaluation(text_raw, questions, min_qual, preferred_qual)
    audio = 0
    video = 0
    
    lex = lexical_eval(text_result)
    score_dict = {"total":total,"text":lex,"audio":audio,"video":video}
    return score_dict



def lexical_eval(text_result):
    min_qual_scores = text_result['minimum_qualification']
    pref_qual_scores = text_result['preferred_qualification']

    min_qual_categories = [categorize_score(score) for score in min_qual_scores]
    pref_qual_categories = [categorize_score(score) for score in pref_qual_scores]
    
    min_avg_score = sum(min_qual_scores) / len(min_qual_scores) if min_qual_scores else 0
    pref_avg_score = sum(pref_qual_scores) / len(pref_qual_scores) if pref_qual_scores else 0
    overall_score = (min_avg_score + pref_avg_score) / 2

    min_avg_category = categorize_score(min_avg_score)
    pref_avg_category = categorize_score(pref_avg_score)
    overall_category = categorize_score(overall_score)

    result_summary = {
        'min_qual_scores': min_qual_categories,
        'pref_qual_scores': pref_qual_categories,
        'min_avg_score': min_avg_category,
        'pref_avg_score': pref_avg_category,
        'overall_score': overall_category
    }

    return result_summary

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
