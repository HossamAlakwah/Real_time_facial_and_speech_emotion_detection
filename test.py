import sounddevice as sd
import numpy as np
import librosa
from keras.models import load_model
import threading

# Load the trained model
model = load_model('speech_emotion_model.h5')

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


# Continuously record and predict audio until 'q' is pressed
while not terminate:
    # Start recording
    print("Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()

    # Extract features from the recorded audio
    mfcc_features = extract_mfcc(audio.flatten(), sample_rate)
    input_data = np.expand_dims(mfcc_features, axis=0)

    # Make a prediction
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions)

    # Print the predicted emotion
    emotions = ['Neutral', 'Angry', 'Happy', 'Sad', 'Fear', 'Disgust', 'Surprise']
    print("Predicted emotion:", emotions[predicted_label])