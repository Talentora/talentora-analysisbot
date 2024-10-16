#https://huggingface.co/facebook/bart-large-cnn

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    max_chunk = 1000  # Adjust based on model's max tokens
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summarized_chunks = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(summarized_chunks)

def dialogue_processing(dialogue):
    for entry in dialogue:
        entry['summarized-answer'] = summarize_text(entry['raw-answer'])
    return dialogue



#def summarize_text(text):
#    summarized_text = summarizer(text, max_length=130, min_length=30, do_sample=False)
#    return summarized_text
