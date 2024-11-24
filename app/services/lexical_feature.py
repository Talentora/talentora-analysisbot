import openai
import re


def total_speech(text_raw):
    speech_script = ""
    for answer in text_raw:
        speech_script += answer
        speech_script += " "
    return speech_script

def extract_score(answer):
    match = re.search(r'\b([1-9][0-9]?|100)\b', answer)
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
            "Provide a score from 1 to 100 based on how well they meet this qualification."
            "\nRespond only with a number from 1 to 100."
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
            "Provide a score from 1 to 100 based on how well they meet this qualification."
            "\nRespond only with a number from 1 to 100."
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
