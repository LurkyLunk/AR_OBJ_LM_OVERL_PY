import os
import cv2
import wikipedia
import tensorflow as tf
import numpy as np
from google.cloud import vision

# Google Cloud Credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r''

# Load TensorFlow pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

def detect_landmarks(frame):
    client = vision.ImageAnnotatorClient()
    _, buffer = cv2.imencode('.jpg', frame)
    image = vision.Image(content=buffer.tobytes())
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    return landmarks

def get_wikipedia_summary(query, sentences=3):
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return summary
    except:
        return None

def preprocess_for_mobilenetv2(image_array):
    resized_image = cv2.resize(image_array, (224, 224))
    processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(resized_image)
    expanded_dims = np.expand_dims(processed_image, axis=0)
    return expanded_dims

def predict_object(image_array):
    preprocessed = preprocess_for_mobilenetv2(image_array)
    predictions = model.predict(preprocessed)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    return decoded_predictions[0][0][1]


def real_time_landmark_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detect_landmarks(frame)
        for landmark in landmarks:
            cv2.putText(frame, landmark.description, (int(landmark.bounding_poly.vertices[0].x), int(landmark.bounding_poly.vertices[0].y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            wiki_summary = get_wikipedia_summary(landmark.description)
            if wiki_summary:
                cv2.putText(frame, wiki_summary, (int(landmark.bounding_poly.vertices[0].x), int(landmark.bounding_poly.vertices[0].y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        obj_detected = predict_object(frame)
        cv2.putText(frame, "Object: " + obj_detected, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow('Real-time Landmark and Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

real_time_landmark_detection()
