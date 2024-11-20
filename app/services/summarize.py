import openai
from app.services.lexical_feature import total_speech

def dialogue_processing(text_raw, questions):
    speech_script = total_speech(text_raw)
    response = summarize_text(speech_script, questions)
    return response

def summarize_text(text, questions): #min_qual is a list whose elements are string
    questions_text = "\n".join([f"Q{i+1}: {question}" for i, question in enumerate(questions)])

    prompt = ("You are an assistant assessing qualifications of an interviewee based on their answers.\n\n"
            f"Questions asked during the interview:\n{questions_text}\n\n"
            f"Transcribed Interview:\n{text}\n\n"
            "Based on the interview questions, summarize the interview response.")
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an assistant assessing interview responses."},
                    {"role": "user", "content": prompt}]
        )

        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error in generating summary: {e}"