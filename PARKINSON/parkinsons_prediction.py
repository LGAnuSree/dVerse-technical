import numpy as np
import pandas as pd
import os
import librosa
import tensorflow as tf
import streamlit as st
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout, MaxPooling1D, BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from math import pi

# 📌 Extract ZIP files if needed
def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

extract_zip("pva_wav_train.zip", "pva_wav_train")
extract_zip("pva_wav_test.zip", "pva_wav_test")

# 📌 Class Label Mapping (Hoehn & Yahr Stages)
class_labels = {
    0: "No Parkinson’s",
    1: "Mild Parkinson’s",
    2: "Moderate Parkinson’s",
    3: "Advanced Parkinson’s",
    4: "Severe Parkinson’s"
}

# 📌 Load and Process CSV Data
def load_and_preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])  
    df.fillna(0, inplace=True)  

    y = df['hoehn_yahr'].values if 'hoehn_yahr' in df.columns else np.zeros(len(df))

    unique_classes = np.unique(y)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
    y = np.array([label_mapping[label] for label in y])

    X = df.drop(columns=['hoehn_yahr'], errors='ignore').values
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y

csv_train_path = "plmpva_train.csv"
csv_test_path = "plmpva_test-WithPDRS.csv"
X_train_csv, y_train = load_and_preprocess_csv(csv_train_path)
X_test_csv, y_test = load_and_preprocess_csv(csv_test_path)

num_classes = len(np.unique(y_train))

# 🎵 Extract MFCC Features from Audio
def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def load_audio_features(audio_folder, num_samples):
    audio_files = os.listdir(audio_folder)[:num_samples]
    return np.array([extract_mfcc(os.path.join(audio_folder, file)) for file in audio_files])

audio_train_path = "pva_wav_train/wav/"
audio_test_path = "pva_wav_test/wav/"

X_train_audio = load_audio_features(audio_train_path, len(X_train_csv))
X_test_audio = load_audio_features(audio_test_path, len(X_test_csv))

X_train = np.hstack((X_train_csv, X_train_audio))
X_test = np.hstack((X_test_csv, X_test_audio))

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 📌 Build CNN + LSTM Model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')  
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and Save the Model
model = build_model((X_train.shape[1], 1), num_classes)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save("parkinsons_model.h5")

# ✅ FIX: Load and Compile Model to Prevent Warning
model = load_model("parkinsons_model.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 🔥 Warning Fix

# ✅ Predict & Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"✅ Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")
print(f"✅ F1 Score: {f1:.2f}")





import streamlit as st
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import wavio
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from math import pi
import tempfile
import os

# 🔑 Configure Gemini API Key
GEMINI_API_KEY = "AIzaSyAF7SLJ_ldhzLOYE-yAZeagEq72eTOmO_k"
genai.configure(api_key=GEMINI_API_KEY)

# 🎭 Sidebar UI
st.sidebar.image("sidebar_logo.png", use_container_width=True)
st.sidebar.title("⚕️ Parkinson's Disease Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Audio", type=["csv", "wav"])

# 🎤 **Real-Time Voice Recording**
def record_audio(duration=5, sr=22050):
    """Records audio from microphone and saves it as a .wav file."""
    st.sidebar.write("🎙️ Recording in progress... Speak Now!")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.int16)
    sd.wait()
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(temp_file.name, recording, sr, sampwidth=2)
    
    st.sidebar.write("✅ Recording complete. Processing...")
    return temp_file.name

if st.sidebar.button("🎤 Record Voice (5s)"):
    recorded_file = record_audio()
    uploaded_file = recorded_file  # Set recorded file as input

# 📌 Class Label Mapping
class_labels = {
    0: "No Parkinson’s",
    1: "Mild Parkinson’s",
    2: "Moderate Parkinson’s",
    3: "Advanced Parkinson’s",
    4: "Severe Parkinson’s"
}

# 🎵 Extract MFCC Features from Audio
def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

# 📌 Load Model
model = load_model("parkinsons_model.h5")

# 🧠 Gemini Explanation Function
def get_gemini_explanation(prediction_label, input_type):
    prompt = f"""
    Based on the diagnosis '{prediction_label}' for Parkinson’s disease using {input_type} data, 
    provide a brief explanation in bullet points. Categorize the response into:
    1. **Causes** (if applicable)
    2. **Lifestyle**
    3. **Treatment**
    4. **Curing Speed**
    5. **Reasons for the Disease**
    6. **Symptoms**

    Each category should have **at most 3 bullet points** and be concise.
    """
    
    try:
        response = genai.GenerativeModel("gemini-1.5-pro-latest").generate_content(prompt)
        if response and hasattr(response, "text"):
            return response.text
        else:
            return "No explanation available."
    except Exception as e:
        return f"Error fetching explanation: {str(e)}"

# 🏥 Main Prediction Processing
if uploaded_file:
    st.title("🩺 Prediction Results")

    if isinstance(uploaded_file, str) or uploaded_file.type == "audio/wav":
        audio_features = extract_mfcc(uploaded_file)
        X_scaled = StandardScaler().fit_transform([audio_features])
        X_scaled = X_scaled.reshape((1, X_scaled.shape[1], 1))

        raw_prediction = model.predict(X_scaled)
        final_prediction = np.argmax(raw_prediction, axis=1)[0]

        class_probs = raw_prediction.flatten()

        prediction_label = class_labels.get(final_prediction, "Unknown")
        explanation = get_gemini_explanation(prediction_label, "Audio")  # ✅ Assign Explanation

    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        df = df.select_dtypes(include=[np.number])
        df.fillna(0, inplace=True)

        X_input = df.values
        X_scaled = StandardScaler().fit_transform(X_input)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

        raw_prediction = model.predict(X_scaled)
        final_prediction = np.argmax(raw_prediction, axis=1)[0]  # Get the first sample's prediction

        class_probs = np.mean(raw_prediction, axis=0)  # Get class probabilities

        prediction_label = class_labels.get(final_prediction, "Unknown")  # Convert to readable label
        explanation = get_gemini_explanation(prediction_label, "CSV")  # ✅ Assign Explanation

    # 🔍 **Final Diagnosis**
    st.subheader("🔍 Final Diagnosis")
    st.write(f"✅ **Prediction: {prediction_label}**")

    # 📊 **Confidence Scores**
    st.subheader("📊 Confidence Scores")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_labels.values(), class_probs, color="skyblue", alpha=0.7)
    ax.set_ylabel("Confidence Score")
    ax.set_title("Model's Confidence per Class")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig)

    # 🤖 **AI Explanation**
    st.markdown("## 🤖 AI Explanation")
    
    if explanation.startswith("Error"):
        st.warning("⚠️ AI explanation is not available at the moment.")
    else:
        st.write(explanation)

  

    # Cleanup temp files
    if isinstance(uploaded_file, str) and os.path.exists(uploaded_file):
        os.remove(uploaded_file)


