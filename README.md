# Sign Language Translator
A real-time sign language translator using Python and OpenCV, built as a university project for Project S2.

## Overview
This project aims to bridge the communication gap between sign language users and non-signers by recognizing static hand gestures from the American Sign Language (ASL) alphabet and translating them into text in real time.

The system is built in three stages:
1. **Data Collection** – Capturing and organizing images for each letter.
2. **Model Training** – Training a classifier on preprocessed hand gesture images.
3. **Gesture Recognition** – Using the trained model to predict hand signs from webcam input.

## Technologies Used
- Python 3
- OpenCV
- NumPy
- Pickle

## How to Run
1. Clone this repo: git clone https://github.com/ani-dilanyan/Sign-Language-Translator.git cd Sign-Language-Translator
3. Install dependencies: pip install -r requirements.txt
4. To collect images: python collect_imgs.py
5. To create and save the dataset: python create_dataset.py
6. To train the model: python train_classifier.py
7. To run real-time gesture recognition: python inference_classifier.py


