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

# **Project Overview**  
### **Data Processing**  
- **Loads & preprocesses `.csv` files** containing patient **biometric data**.  
- **Extracts MFCC features** from `.wav` audio files to analyze speech variations.  
- **Standardizes all features** using **`StandardScaler()`** to optimize model performance.  

# **Model Architecture (CNN + LSTM)**  
The model consists of:  
- **1D Convolutional Layers** → Extracts speech frequency patterns.  
- **Batch Normalization & MaxPooling** → Enhances feature learning.  
- **LSTM Layers** → Captures time-dependent speech irregularities.  
- **Dense Layers** → Fully connected layers for classification.  
- **Softmax Activation** → Predicts **Hoehn & Yahr severity stages** (0 to 4).
-----------------------------------------------------------------------------------------------

# **2. Hand Landmark Detection Using Computer Vision**  
*A Real-Time AI System for Hand Tracking & Gesture Recognition*  

This project implements **real-time hand tracking** using **Computer Vision (OpenCV) and AI (MediaPipe)**. The system detects **hand landmarks** (e.g., fingertips, joints) and visualizes them in a live camera feed.  

## **Features**  
- **Real-Time Hand Tracking** → Detects and tracks hands using a webcam.  
- **Landmark Detection** → Identifies **21 hand key points** (e.g., fingertips, joints).  
- **Fast & Efficient Model** → Powered by **MediaPipe**, optimized for real-time performance.  
- **Dynamic Landmark Indexing** → Labels each landmark for easy analysis.  
- **Flexible Input** → Works with both **built-in and external webcams**.  
- **Customizable Confidence Levels** → Adjustable detection & tracking thresholds.  


## **Project Overview**  
### **Hand Detection & Tracking**  
- Uses **MediaPipe Hands Model** to detect **hand presence** in a video feed.  
- Captures **21 key landmark points** for each detected hand.  
- Converts landmark positions from **normalized coordinates** to **pixel values**.  

### **Landmark Visualization**  
- Uses **OpenCV** to draw **landmarks & connections** on detected hands.  
- **Displays landmark indices** to analyze hand structure.  

### **Gesture Recognition (Future Work)**  
- Can be extended to recognize **specific hand gestures** (e.g., sign language, counting fingers).  
- Potential applications in **AR/VR, gaming, and human-computer interaction**.  


## **🛠️ Technologies Used**  
- **Computer Vision:** OpenCV  
- **Hand Tracking AI Model:** MediaPipe Hands  
- **Programming Language:** Python.
-----------------------------------------------------------------------------------------------



