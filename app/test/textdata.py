from app.test.lexical_feature import text_evaluation
from app.services.score_calculation import lexical_eval

question1 = "Can you introduce yourself"
question2 = "Why did you apply to this position"
question3 = "Explain about your work experience in a team."
question4 = "Describe a challenge you faced in your last job and how you overcame it."
question5 = "What is your strength and weakness"
questions = [question1, question2, question3, question4, question5]
#good answer
goodAnswer = [
    "I have over five years of experience in software engineering, with a strong background in backend development. I'm passionate about creating efficient solutions and always eager to learn and take on new challenges.",
    "I applied to this position because I'm excited about the opportunity to work on innovative projects and contribute my skills in a collaborative environment. I also admire the company's commitment to growth and am eager to be part of that journey.",
    "In my previous role, I collaborated closely with cross-functional teams, including designers and QA testers, to deliver high-quality features. I value teamwork and actively seek feedback to ensure our project goals are met.",
    "In my last job, I encountered a major project delay due to a team member's unexpected leave. I took the initiative to reallocate tasks among the team, coordinating daily check-ins, and we were able to meet our deadline.",
    "My strength lies in my analytical skills and attention to detail, which helps me solve complex issues efficiently. My weakness is that I can sometimes focus too much on minor details, but I've been working on improving my time management to balance priorities better."
]


#bad answer
badAnswer = [
    "I'm just a regular person. Not much to say about myself.",
    "I applied because I need a job and this one seems okay.",
    "I’ve worked with other people before, but I prefer working alone most of the time.",
    "I had issues in my last job, but I don’t want to talk about it. It was mostly my manager’s fault.",
    "I’m really good at everything, so I don’t have any weaknesses."
]


min_qual = ["teamwork experience","descriptive answer","solved technical problem","coherency","technical skills"]
pref_qual = ["project management experience","willingness to learn","communication skills","test driven development","startup experience"]

good_result = lexical_eval(goodAnswer, questions, min_qual, pref_qual)
bad_result = lexical_eval(badAnswer, questions, min_qual, pref_qual)

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