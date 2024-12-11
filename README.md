Fall Detection Using Deep Learning

Overview

This project leverages deep learning techniques to detect falls in elderly individuals, aiming to enhance safety and provide timely assistance. The system combines human pose estimation using PoseLandmark with a Long Short-Term Memory (LSTM) neural network to analyze sequential data and predict fall events with high accuracy.

Features

Pose Estimation: Extracts key body landmarks using PoseLandmark to represent human poses as matrices.

LSTM Neural Network: Processes the time-series data from PoseLandmark matrices to identify patterns associated with falls.

High Accuracy: Achieved 90% accuracy in detecting falls during testing.


  

Usage
1. Pose Extraction
Use the scripts in extract_pose/ to preprocess video data and generate PoseLandmark matrices.

2. Train the Model
Train the LSTM model using:

bash
Copy code
python training_model.py
The trained model will be saved in the model/ directory.

3. Evaluate the Model
Evaluate the model's performance:

bash
Copy code
python evaluate_metrics.py
4. Real-Time Detection
Run the detection system:

bash
Copy code
python main.py
Results
Accuracy: 90%
Model Architecture:
PoseLandmark data: 33 body landmarks.
LSTM layers: Two hidden layers with 128 units.
Technologies Used
Python
TensorFlow/Keras
OpenCV
Mediapipe for pose extraction.
Contact
Feel free to reach out for feedback or contributions:

Name: Tran Tien Dat
Email: trantiendatdn2004@gmail.com
