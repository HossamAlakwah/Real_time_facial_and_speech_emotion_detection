import sounddevice as sd
import numpy as np
import librosa
from keras.models import load_model
import threading
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained models
speech_model = load_model('speech_emotion_model.h5')
emotion_model = tf.keras.models.load_model('model.h5')

# Function to extract MFCC features from audio data
def extract_mfcc(audio_data, sample_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfcc

# Set the duration of each audio recording in seconds
duration = 5

# Set the sample rate for audio recording
sample_rate = 22050

# Flag to check if 'q' has been pressed
terminate = False

# Global variable to store speech emotion
speech_emotion = None

# Lock for synchronizing access to speech emotion
speech_lock = threading.Lock()

# Function for speech emotion detection
def speech_emotion_detection():
    global terminate, speech_emotion

    while not terminate:
        # Start recording
        print("Recording audio...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        # Extract features from the recorded audio
        mfcc_features = extract_mfcc(audio.flatten(), sample_rate)
        input_data = np.expand_dims(mfcc_features, axis=0)

        # Make a prediction
        predictions = speech_model.predict(input_data)
        predicted_label = np.argmax(predictions)

        # Store the predicted emotion
        emotions = ['Neutral', 'Angry', 'Happy', 'Sad', 'Fear', 'Disgust', 'Surprise']
        with speech_lock:
            speech_emotion = emotions[predicted_label]

        # Print the predicted emotion
        print("Predicted emotion (Speech):", speech_emotion)

# Class for facial expression recognition
class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model):
        self.model = model

    def predict_emotion(self, img):
        self.preds = self.model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = FacialExpressionModel(emotion_model)
font = cv2.FONT_HERSHEY_SIMPLEX

# Class for video camera
class VideoCamera(object):
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = emotion_model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0, 0), 2)
            return fr
def gen(camera):
    while not terminate:
        with speech_lock:
            speech_text = speech_emotion
        frame = camera.get_frame()
        cv2.putText(frame, f"Speech Emotion: {speech_text}", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.imshow('Facial Expression Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

video_camera = VideoCamera(0) # Use 0 for the default camera, or specify the path to a video file
speech_thread = threading.Thread(target=speech_emotion_detection)
video_thread = threading.Thread(target=gen, args=(video_camera,))
video_thread.start()
speech_thread.start()
video_thread.join()
speech_thread.join()
terminate = True

