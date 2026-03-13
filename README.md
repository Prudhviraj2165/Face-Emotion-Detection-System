# 👁️ FeelCam: AI-Powered Face Emotion Detection System

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Prudhviraj2165/Face-Emotion-Detection-System)
[![Python Version](https://img.shields.io/badge/Python-3.8+-yellow?logo=python)](https://www.python.org/)
[![Flask Framework](https://img.shields.io/badge/Framework-Flask-lightgrey?logo=flask)](https://flask.palletsprojects.com/)

> **[🚀 Access Project Repository](https://github.com/Prudhviraj2165/Face-Emotion-Detection-System)**

FeelCam is a modern, real-time emotion recognition web application. It uses Deep Learning to analyze facial expressions through live camera feeds or static image uploads, providing detailed emotional health insights and personalized recommendations.

![Dashboard Preview](static/images/hero-bg.jpg) <!-- Placeholder for actual screenshot if available -->

## 🚀 Features

- **Real-time Video Analysis:** Scan your face live and see emotional shifts in real-time.
- **Image Upload:** Upload photos for instant emotional breakdown.
- **Modern Dashboard:** Track your emotional history and view trends.
- **Interactive Charts:** Beautiful horizontal bar graphs powered by Chart.js visualize emotion distribution.
- **Smart Recommendations:** Receive tailored suggestions and podcast links based on your detected mood.
- **Robust Detection:** Powered by OpenCV for reliable face tracking across all environments.

## 🛠️ Technology Stack

- **Backend:** Flask (Python)
- **Database:** SQLite (SQLAlchemy)
- **Machine Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV (Haar Cascades)
- **Frontend:** HTML5, CSS3 (Vanilla), JavaScript
- **Visualization:** Chart.js

## 📥 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Prudhviraj2165/Face-Emotion-Detection-System.git
   cd Face-Emotion-Detection-System
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv env
   source env/bin/scripts/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Model:**
   If you don't have the pre-trained model, run the download script:
   ```bash
   python download_model.py
   ```

## 🎮 How to Run

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   `http://127.0.0.1:5000`

3. **Register/Login** to access the dashboard and analysis tools.

## 📉 ML Pipeline

The system uses a Convolutional Neural Network (CNN) trained on the **FER-2013** dataset. 
- **Preprocessing:** Images are converted to grayscale and normalized using CLAHE.
- **Augmentation:** Test-Time Augmentation (TTA) is used during inference for higher stability.
- **Thresholding:** Confidence scores below 25% are rejected to ensure accuracy.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
