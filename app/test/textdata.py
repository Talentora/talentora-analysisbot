from services.lexical_feature import text_evaluation
#behavioral and social skills
from configs import openai_config
openai_config.configure_openai("API KEY")

question1 = "How did you find job opening"
question2 = "Can you introduce yourself"
question3 = "Why did you apply to this position"
question4 = "Explain about your work experience in a team"
question5 = "Describe a challenge you faced in your last job and how you overcame it."
#good answer
goodAnswer = [
    "I found the job opening through LinkedIn and was intrigued by the position.",
    "I am a software engineer with 5 years of experience in backend development.",
    "I applied because I align with the company's mission and values.",
    "In my previous role, I collaborated with a cross-functional team to build a scalable product.",
    "During a project deadline, I coordinated with my team to manage and solve unexpected issues."
]



#bad answer
badAnswer = [
    "I just saw it online.",
    "I'm a person who likes coding.",
    "I need a job.",
    "Worked in a team before.",
    "Had some issues but managed."
]




good_result = text_evaluation(goodAnswer)
bad_result = text_evaluation(badAnswer)

print("Good Answers Evaluation:")
print(good_result)
print()
print("Bad Answers Evaluation:")
print(bad_result)






"""
from summarize import dialogue_processing
#get json type text data first


#store the data in pandas DataFrame
d = {'question':questions,'raw-answer':raw_answers}
dialogue = [{'question': q, 'raw-answer': a} for q, a in zip(questions, raw_answers)]

dialogue_processing(dialogue)

# Print the results
for entry in dialogue:
    print(f"Question: {entry['question']}")
    print(f"Raw Answer: {entry['raw-answer']}")
    print(f"Summarized Answer: {entry['summarized-answer']}\n")
    print('---')


#fake transcript
#try summarization with 
"""