import cv2
import dlib
import numpy as np
from imutils import face_utils

# initialize dlib's face detector (HOG-based) and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

# open the default webcam (usually the first webcam)
cap = cv2.VideoCapture(0)  # ONLY change the index if we want to test multiple cameras

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # resize the frame for faster processing (optional, subject to change)
    frame = cv2.resize(frame, (640, 480))
    
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame (face detection algorithms often perform better and faster on grayscale images)
    # second argument is upsample factor. 0 means no upsampling. Increasing this value can help detect smaller faces at cost of speed
    rects = detector(gray, 0)

    # loop over the face detections (for multiple faces)
    for rect in rects:
        # get facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # converts the dlib rectangles to a NumPy array of (x, y) coordinates

        # draw face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # draw facial landmarks
        for (x_point, y_point) in shape:
            cv2.circle(frame, (x_point, y_point), 1, (0, 0, 255), -1)

        # perform specific analyses
        # emotion Detection, Mouth Movement, Eye Tracking (implement functions below)
        
        # extract the face region of interest (ROI) -- example below:
        # face_roi = gray[y:y + h, x:x + w]
        # emotion = detect_emotion(face_roi)
        # cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        # mouth movement analysis -- example below: 
        # mouth_status = analyze_mouth(shape)
        # cv2.putText(frame, mouth_status, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # eye tracking -- example below:
        # eye_status = analyze_eyes(shape)
        # cv2.putText(frame, eye_status, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)



    # display the frame
    cv2.imshow("Real-Time Facial Analysis", frame)

    # break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera and close windows
cap.release()
cv2.destroyAllWindows()

def detect_emotion(face_roi):
    # preprocess the face ROI as required by model
    # for example, resize, normalize, and reshape

    # predict emotion using (tbd) model i.e.:
    # emotion_label = model.predict(processed_roi)

    # return emotion label
    return emotion_label

def analyze_mouth(shape):
    # mouth landmarks are points 48-67 (for shape predictor 68)
    mouth_points = shape[48:68]
    # compute mouth aspect ratio or other features
    # implement logic or model to detect speech, yawning, etc.

    # return analysis result
    return mouth_status

def analyze_eyes(shape):
    # left eye landmarks are points 36-41 (for shape predictor 68)
    # right eye landmarks are points 42-47 (for shape predictor 68)
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    # calculate eye aspect ratio or gaze direction
    # implement logic to detect blinks, gaze direction, etc.

    # return analysis result
    return eye_status

