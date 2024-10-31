"""
analyzing what the candidate is saying and if their responses align with a good answer 
for that role and company
"""
from nltk.tokenize import word_tokenize 
import spacy
import openai

def lexical_feature_eval(text_raw):
    lex_analysis_data = qualification(text_raw)

    return lex_analysis_data

def total_speech(text_raw):
    speech_script = ''
    for i in range(len(text_raw)):
        speech_script += text_raw[i]
    return speech_script

def qualification(text_raw):
    result = {'minimum qualification','preferred_qualification','answer_quality'}
    speech_script = total_speech(text_raw)
    minimum_qualification(speech_script)
    preferred_qualification(speech_script)
    answer_quality(speech_script)
    return result

def minimum_qualification(text):
    result = {}
    min_qual = ["No"] * 5  # Assuming a max of 5 qualifications

    for i in range(len(min_qual)):
        response = openai.ChatCompletion.create(
            model="gpt-3.5",
            messages=[
                {"role": "system", "content": "You are an assistant assessing qualifications."},
                {"role": "user", "content": f"In the interview script, does this person mention content aligned with qualification {i+1}? Answer yes or no. Text: {text}"}
            ]
        )
        answer = response['choices'][0]['message']['content'].strip().lower()
        min_qual[i] = "Yes" if "yes" in answer else "No"

    result['min_qualifications'] = min_qual
    return result

def preferred_qualification(text):
    #max 5
    result = {}
    preferred_qual = ["No"] * 5  # Assuming a max of 5 qualifications

    for i in range(len(preferred_qual)):
        response = openai.ChatCompletion.create(
            model="gpt-3.5",
            messages=[
                {"role": "system", "content": "You are an assistant assessing qualifications."},
                {"role": "user", "content": f"In the interview script, does this person mention content aligned with preferred qualification {i+1}? Answer yes or no. Text: {text}"}
            ]
        )
        answer = response['choices'][0]['message']['content'].strip().lower()
        preferred_qual[i] = "Yes" if "yes" in answer else "No"

    result['preferred_qualifications'] = preferred_qual
    return result

def answer_quality(text):
    result = {}
    #descriptive, coherent
    return result



def sentimental_analysis(text):
    #hume 
    return None

#high wpsec, wc, uc --> better candidate
#low fpsec, filler words, non-fluency words, unvoiced region in speech


#print(sentimental_analysis("I had a difficulty with time management."))
#print(sentimental_analysis("I had a difficulty with time management, but I could overcome this problem with google calendar management"))


"""


def verbal_skill(text,interview_length):
    verbal = {}
    #tokenize
    tokenized_text = word_tokenize(text)
    #data cleaning
    processed_text = ''
    verbal['wpsec'] = len(processed_text)/interview_length
    verbal['upsec'] = 10
    verbal['wc'] = 10
    verbal['uc'] = 10
    verbal['fpsec'] = 10
    return verbal


def pro_count(text):
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    pro_list = {}

    for token in doc:
        if token.pos_ == "PRON":
            pro = token.text
            #print(token.text, token.dep_, token.head.text)
            if pro not in pro_list:
                pro_list[pro] = 1
            else:
                pro_list[pro] += 1

    return pro_list

def lexicon_analysis(text_raw):
    speech_script = total_speech(text_raw) #string type. full interview script

    sentiment = sentimental_analysis(speech_script) #positive/negative/neutral
    pronoun = pro_count(speech_script) #list of pronoun
    #coherence

    speaking_skills = verbal_skill(speech_script, 1800) #30 minutes

    lex_analysis = {}

    return lex_analysis
"""