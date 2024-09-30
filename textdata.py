from summarize import summarize_text
import pandas as pd


#get text data first

#transform text data into dialogue format
questions = []
raw_answers = []


#store the data in pandas DataFrame
d = {'question':questions,'raw-answer':raw_answers}
dialogue = pd.DafaFrame(data=d)

#apply summarization for the each answer
dialogue['summarized-answer'] = dialogue['raw-answer'].apply(summarize_text)

#send data back