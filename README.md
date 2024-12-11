# Fall Detection Using Deep Learning

## Overview
This project leverages deep learning techniques to detect falls in elderly individuals, aiming to enhance safety and provide timely assistance. The system combines human pose estimation using **PoseLandmark** with a **Long Short-Term Memory (LSTM)** neural network to analyze sequential data and predict fall events with high accuracy.

## Features
- **Pose Estimation**: Extracts key body landmarks using PoseLandmark to represent human poses as matrices.
- **LSTM Neural Network**: Processes the time-series data from PoseLandmark matrices to identify patterns associated with falls.
- **High Accuracy**: Achieved 90% accuracy in detecting falls during testing.



## Getting Started

### 1. Data Preparation
- Collect video data of human activities, including falls and non-falls.
- Use **PoseLandmark** to extract body landmarks and save them as matrices.
- Store the data in the `data/` directory.

### 2. Training the Model
Run the training script to train the LSTM model:

```bash
python training_model.py
```

This will:
- Load the preprocessed PoseLandmark data.
- Train the LSTM model.
- Save the trained model in the `model/` directory.

### 3. Testing the Model
Evaluate the model on test data using:

```bash
python evaluate_metrics.py
```

This script will output performance metrics such as accuracy and confusion matrix.

### 4. Real-Time Fall Detection
Run the real-time fall detection script:

```bash
python main.py
```

This script processes live video feed, extracts PoseLandmark features, and predicts falls in real-time.

## Results
- **Accuracy**: 90%
- **Model Architecture**:
  - PoseLandmark feature extraction: 33 body landmarks.
  - LSTM: 2 hidden layers with 128 units each.

## Dataset
The dataset includes video sequences of human activities annotated for falls and non-falls. PoseLandmark matrices are generated from these videos.

## Technologies Used
- **Python**
- **TensorFlow/Keras** for deep learning
- **OpenCV** for video processing
- **Mediapipe** for PoseLandmark extraction

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to the branch.
5. Submit a pull request.

## Contact
For any questions or feedback, feel free to reach out:

- **Name**: trantiendat
- **Email**: trantiendatdn2004@gmail.com


---

Thank you for exploring this project! Together, let's make fall detection systems more reliable and accessible.

