# dVerse-technical
This Repository contains three tasks and their complete implementation done using python. The tasks are as follows: Parkinsons' disease prediction, Detecting hand joints using Computer Vision and a chatbot for FAQ.

# 1. Parkinson’s Disease Detection Using AI  

This project leverages **Deep Learning and Speech Processing** to detect **early signs of Parkinson’s disease** based on **biometric data and speech patterns**. The model combines **CNN + LSTM networks** with **Google Gemini AI** for explainability.  

# **Features**  
- **Multi-Input Analysis** → Supports **`.wav` speech recordings & `.csv` biometric data**.  
- **Speech Feature Extraction** → Uses **MFCCs (Mel-Frequency Cepstral Coefficients)** for precise voice analysis.  
- **Hybrid Deep Learning Model** → **CNN + LSTM** for advanced time-series analysis.  
- **Real-Time Voice Recording** → Record & analyze speech within the app.  
- **AI-Generated Explanations** → **Google Gemini AI** provides insights into predictions.  
- **Confidence Score Visualization** → Displays probability distribution of Parkinson’s severity.  

# **📜 Project Overview**  
### **1️⃣ Data Processing**  
- **Loads & preprocesses `.csv` files** containing patient **biometric data**.  
- **Extracts MFCC features** from `.wav` audio files to analyze speech variations.  
- **Standardizes all features** using **`StandardScaler()`** to optimize model performance.  

### **2️⃣ Model Architecture (CNN + LSTM)**  
The model consists of:  
- **1D Convolutional Layers** → Extracts speech frequency patterns.  
- **Batch Normalization & MaxPooling** → Enhances feature learning.  
- **LSTM Layers** → Captures time-dependent speech irregularities.  
- **Dense Layers** → Fully connected layers for classification.  
- **Softmax Activation** → Predicts **Hoehn & Yahr severity stages** (0 to 4).  


