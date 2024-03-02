import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import linear_model
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from sklearn import metrics
import scipy

EMOTIONS_LIST = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

json_file = open('ved_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
ved_model = model_from_json(loaded_model_json)

ved_model.load_weights("ved_model.h5")
#print("Loaded model from disk")
#path = os.getcwd()

cap = cv2.VideoCapture(r"C:\Users\BIKASH CHANDRA SAHAN\OneDrive\Desktop\VideoStream Emotion Detection\video4.mp4")
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280,720))
    #height,width = frame.shape[:2]
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)


    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)), -1), 0)


        emotion_prediction = ved_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, EMOTIONS_LIST[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)


    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
