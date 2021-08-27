import keras_applications.mobilenet
from imports import *
import cv2
from tensorflow.keras.models import load_model

model = load_model('../models/mobile_net_sign_language_model2.h5')

cap = cv2.VideoCapture(0)

while(True):

    _, frame = cap.read()

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    # Preprocess the frame to predict
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis = 0)  # we need the fourth dimension
    frame_processed = tf.keras.applications.mobilenet.preprocess_input(frame)

    # Predict with the model
    predictions = model.predict(x = frame_processed)

    # Print the prediction
    # print(predictions)
    print(np.argmax(predictions))
