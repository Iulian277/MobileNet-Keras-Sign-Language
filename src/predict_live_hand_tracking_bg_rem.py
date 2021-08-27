import cv2
import mediapipe as mp # For hand tracking
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
from imports import *
from tensorflow.keras.models import load_model


model = load_model('../models/mobile_net_sign_language_model2.h5')
cap = cv2.VideoCapture(0)

# Initialize the hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands = 1)
mpDraw = mp.solutions.drawing_utils # For drawing the points

# Selfie Segmentation
segmentor = SelfiSegmentation()

while True:
    _, img = cap.read()

    # Convert from BGR (cv2 reads) to RGB (Hands() uses)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # There is at least one hand on the frame
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Determine min_x, max_x, min_y, max_y
            x = []
            y = []
            for id, lm in enumerate(handLms.landmark): # For multiple hands in the frame
                # print(id, lm)
                height, width, ch = img.shape
                pos_x, pos_y = int(lm.x * width), int(lm.y * height)
                # print(id, pos_x, pos_y)
                x.append(pos_x)
                y.append(pos_y)

            padding = 40
            min_x = max(0, min(x) - padding)
            max_x = max(x) + padding
            min_y = max(0, min(y) - padding)
            max_y = max(y) + padding

            # Preprocess the frame to predict
            roi = img[min_y: max_y, min_x: max_x]
            # Remove bg
            roi = cv2.resize(roi, (224, 224))
            bg = cv2.imread('../data/bg.png')
            roi = segmentor.removeBG(roi, bg, threshold = 0.65)

            roi = np.expand_dims(roi, axis = 0)  # We need fourth dimensions
            frame_processed = tf.keras.applications.mobilenet.preprocess_input(roi)

            predictions = model.predict(x = frame_processed)

            # Draw the connections between points and a rectangle
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # cv2.rectangle(img, (min_x, min_y), (max_x, max_y ), (255, 0, 0), 2)

            # Print the prediction
            print(np.argmax(predictions))


    cv2.imshow('Video', img)
    cv2.waitKey(1)
