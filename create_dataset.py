import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure hand detection model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define dataset directory
DATA_DIR = './data'

# Lists to store processed hand landmark data and labels
data = []
labels = []

# Iterate through each class folder in the dataset directory
for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)

    # Skip if it's not a directory
    if not os.path.isdir(class_path):
        continue

    # Iterate through each image in the class folder
    for img_path in os.listdir(class_path):
        data_aux = []  # Store extracted hand landmark features
        x_ = []  # Store x-coordinates of hand landmarks
        y_ = []  # Store y-coordinates of hand landmarks

        # Read image and convert to RGB for MediaPipe processing
        img = cv2.imread(os.path.join(class_path, img_path))

        # Ensure image is read correctly before processing
        if img is None:
            continue  

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image to detect hand landmarks
        results = hands.process(img_rgb)
        
        # If hand landmarks are detected, extract them
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect all x, y coordinates of detected hand landmarks
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize landmarks by shifting them relative to the minimum x and y values
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))  # Normalize x
                    data_aux.append(landmark.y - min(y_))  # Normalize y

            # Append the extracted landmark data and class label
            data.append(data_aux)
            labels.append(dir_)


# Save processed data into a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset processing complete. Data saved as 'data.pickle'.")
