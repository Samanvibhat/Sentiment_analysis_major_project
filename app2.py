import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# Load your trained model
model_path = '/content/Sentiment_analysis_major_project/Sentiment_analysis_major_project/my_model.h5'  # Update with your model path
model = load_model(model_path)

# Load the label encoder to map encoded labels back to emotions
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to extract features from audio file
def extract_features(audio_path, duration=3, n_mfcc=13):
    wave, sr = librosa.load(audio_path, duration=duration)
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc)
    
    # Ensure the number of frames matches the expected input shape
    # If the number of frames is less than 13, pad with zeros
    if mfcc.shape[1] < 13:
        mfcc = np.pad(mfcc, ((0, 0), (0, 13 - mfcc.shape[1])), mode='constant')
    # If the number of frames is greater than 13, truncate
    elif mfcc.shape[1] > 13:
        mfcc = mfcc[:, :13]
    
    return mfcc

# Streamlit app
st.title('Emotion Detection from Audio')

# File uploader for audio
uploaded_file = st.file_uploader("Upload an audio file...", type=["wav"])

if uploaded_file is not None:
    # Preprocess the uploaded audio file
    audio_path = 'temp_audio.wav'  # Temporary path for the uploaded file
    with open(audio_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Extract features from the uploaded audio
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=-1)  # Add channel dimension
    
    # Predict emotion
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    st.write("Predicted Emotion:", predicted_label)
