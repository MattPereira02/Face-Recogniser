# Age, Gender & Emotion Recognition App

An Android application that uses TensorFlow Lite (TFLite) and ML Kit Face Detection to recognize age, gender, and facial expression in real-time or from images. The app demonstrates lightweight deep learning model deployment on mobile devices, with a focus on edge performance and usability.

## Features

- Select an image from the gallery for analysis.
- Detect faces using Google ML Kit.
- Predict emotion (happy, sad, etc.).
- Predict gender (male, female).
- Estimate age (regression model).
- Optimized for real-time performance on edge devices using TFLite.

## Tech Stack

- Android Studio (Java)
- TensorFlow Lite (for inference)
- ML Kit Face Detection (for face localization)
- Pretrained CNN models for age, gender, and emotion

## Project Structure
```
app/
 ├── java/com/example/facerecogniser/
 │    ├── MainActivity.java       # App entry point
 │
 ├── assets/
 │    ├── age_model.tflite        #Trained models
 │    ├── gender_model.tflite
 │    └── emotion_model.tflite
 │
 └── res/layout/
      └── activity_main.xml       # UI layout
```
## Setup & Installation

### Clone repository
```
git clone https://github.com/yourusername/facerecogniser.git
cd facerecogniser
```
### Open in Android Studio

- Select "Open an Existing Project"
- Choose the facerecogniser folder

### Add models

- Place the following .tflite models in app/src/main/assets/
  - age_model.tflite
  - gender_model.tflite
  - emotion_model.tflite

### Build & Run

- Connect an Android device
- Click Run in Android Studio

## Performance

| Model         | Input Shape | Avg. Inference Time (ms) | Accuracy (Test Set) |
| ------------- | ----------- | ------------------------ | ------------------- |
| Emotion Model | 64×64×3     | \~3.1 ms                 | \~80%               |
| Gender Model  | 128×128×3   | \~1.6 ms                 | \~85%               |
| Age Model     | 200×200×3   | \~4.6 ms                 | MAE ≈ 9.5 years     |

## License

This project is licensed under the MIT License.

