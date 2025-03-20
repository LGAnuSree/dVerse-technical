import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = "parkinson_model.h5"

try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    st.error("Model file not found! Please train the model first.")
    model_loaded = False

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=150):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Padding or truncating the MFCC sequence to fixed length
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return np.array(mfccs)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Streamlit UI
st.title("Parkinson’s Disease Detection from Speech")
st.write("Upload a **WAV** file to analyze its MFCC features and detect Parkinson's Disease.")

# File Upload Section
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_path = f"temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File uploaded successfully: {uploaded_file.name}")

    # Extract MFCC features
    mfcc_features = extract_features(file_path)

    if mfcc_features is not None:
        # Display waveform and MFCC features
        fig, ax = plt.subplots(figsize=(8, 4))
        librosa.display.specshow(mfcc_features, x_axis='time', cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title("MFCC Features")
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficients")
        st.pyplot(fig)

        if model_loaded:
            # Reshape input for model prediction
            mfcc_features = mfcc_features.reshape(1, mfcc_features.shape[0], mfcc_features.shape[1])

            # Predict
            prediction = model.predict(mfcc_features)
            class_labels = ["Healthy", "Parkinson's"]
            result = class_labels[np.argmax(prediction)]

            st.subheader(f"🩺 **Prediction: {result}**")
            st.write(f"Confidence: {prediction[0][np.argmax(prediction)]:.2f}")

            if result == "Parkinson's":
                st.warning("High risk of Parkinson's detected. Consult a doctor for further evaluation.")
            else:
                st.success("No signs of Parkinson's detected.")