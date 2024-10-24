import video
import audio
import summarize

def score():
    #input: 
    #output: dictionary of scores (total, individual scores from different eval method)
    total = 0
    text = 0
    audio = 0
    video = 0
    #exact variable needs to be discussed
    score_dict = {"total":total,"text":text,"audio":audio,"video":video}
    return score_dict