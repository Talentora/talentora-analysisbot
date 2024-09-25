#https://huggingface.co/facebook/bart-large-cnn


from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(text):
    summarizer(text, max_length=130, min_length=30, do_sample=False)
