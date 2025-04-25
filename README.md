# Interview Analysis Bot

An intelligent application designed to analyze and evaluate interview transcripts using AI. This tool helps assess interview quality and provides insights into candidate responses.

## 🚀 Features

- Automated interview transcript analysis
- Real-time processing of interview data
- Support for different interview positions and durations
- RESTful API endpoints for integration
- Structured evaluation of candidate responses

## 📋 Project Structure

## Run instructions
1. Run docker
2. Run this command:
```
docker build -t talentora-analysisbot .
docker run -d -p 8000:8000 --name talentora talentora-analysisbot
```
3. Now run localhost:8000 to verify
4. To stop, simply run 
```
docker stop talentora
```
5. meow

## Input

1) Video: Interview Recording

2) Text: Transcribed Interview Content 

3) Audio: Extracted Audio File from Interview Recording

## Output

Evaluate Hard skill, Soft skill, Culture fit

1) Summmarize Key points of interview

What do Recruiter exactly want to know? What kinds of keypoints?git config pull.rebase true

[Question, Answer Summary]

2) Similar Goal share

-hard skill fit: list of skills required for the position

-soft skill fit

-culture fit

3) Emotion/ speech sentimental analysis

4) Tone/ Pitch Analysis

5) Mouth movement/ Eye tracking

# Models (IN TESTING PHASE)

## videodata.py 

### Face Detection with dlib
- **Algorithm Used**: Histogram of Oriented Gradients (HOG) combined with a Linear Support Vector Machine (SVM).
  - The image is scanned at multiple scales.
  - HOG features are extracted at each scale.
  - The Linear SVM classifier predicts whether a region contains a face.

### Facial Landmark Detection
- **Predictor Model**: shape_predictor_68_face_landmarks.dat
  - A pre-trained model that can predict 68 landmark points on a human face.
- **Landmarks**: 
  - Points 1-17: Contour of the face
  - Points 18-27: Eyebrows
  - Points 28-36: Nose
  - Points 37-48: Eyes
  - Points 49-68: Mouth

### TO DO
- [ ] Emotion Detection
- [ ] Mouth Movement Analysis
- [ ] Eye Tracking and Blink Detection
