"""
analyzing what the candidate is saying and if their responses align with a good answer 
for that role and company
"""
from nltk.tokenize import word_tokenize 
import openai
import re


def total_speech(text_raw):
    speech_script = ""
    for answer in text_raw:
        speech_script += answer
        speech_script += " "
    return speech_script

def extract_score(answer):
    match = re.search(r'\b([1-9]|10)\b', answer)
    if match:
        return int(match.group(1))
    else:
        return 0  # Default to 0 if score is not found

def text_evaluation(text_raw, questions, min_qual, preferred_qual):
    speech_script = total_speech(text_raw)
    result = {
        'minimum_qualification': minimum_qualification(speech_script, questions, min_qual),
        'preferred_qualification': preferred_qualification(speech_script, questions, preferred_qual)
    }
    return result

def minimum_qualification(text, questions, min_qual): #min_qual is a list whose elements are string
    result = [0] * len(min_qual)

    questions_text = "\n".join([f"Q{i+1}: {question}" for i, question in enumerate(questions)])

    for i, qualification in enumerate(min_qual):
        # Include the questions, interview transcript, and specific qualification
        prompt = (
            "You are an assistant assessing qualifications of an interviewee based on their answers.\n\n"
            f"Questions asked during the interview:\n{questions_text}\n\n"
            f"Transcript:\n{text}\n\n"
            f"Evaluate the interviewee's qualification for: '{qualification}'.\n"
            "Provide a score from 1 to 10 based on how well they meet this qualification."
            "\nRespond only with a number from 1 to 10."
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant assessing interview responses."},
                {"role": "user", "content": prompt}
            ]
        )
        #parsing
        answer = response['choices'][0]['message']['content'].strip()
        score = extract_score(answer)
        result[i] = score

    return result

def preferred_qualification(text, questions, preferred_qual):
    result = [0] * len(preferred_qual)

    questions_text = "\n".join([f"Q{i+1}: {question}" for i, question in enumerate(questions)])

    for i, qualification in enumerate(preferred_qual):
        # Include the questions, interview transcript, and specific preferred qualification
        prompt = (
            "You are an assistant assessing the preferred qualifications of an interviewee based on their responses.\n\n"
            f"Questions asked during the interview:\n{questions_text}\n\n"
            f"Transcript:\n{text}\n\n"
            f"Evaluate the interviewee's alignment with the preferred qualification: '{qualification}'.\n"
            "Provide a score from 1 to 10 based on how well they meet this qualification."
            "\nRespond only with a number from 1 to 10."
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant assessing interview responses."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse
        answer = response['choices'][0]['message']['content'].strip()
        score = extract_score(answer)
        result[i] = score

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