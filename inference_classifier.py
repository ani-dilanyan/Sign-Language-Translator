import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load trained model
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize OpenCV video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Define label mapping
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z mapping

# Define custom colors for landmarks and connections
landmark_color = (184, 57, 120)       # Purple
connection_color = (214, 198, 56)     # Yellow-ish

while True:
    # Auxiliary data lists
    data_aux = []
    x_ = []
    y_ = []

    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(frame_rgb)

    # If landmarks detected
    if results.multi_hand_landmarks:
        # Use only the first hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        landmark_drawing_spec = mp_drawing.DrawingSpec(color=landmark_color, thickness=3, circle_radius=3)
        connection_drawing_spec = mp_drawing.DrawingSpec(color=connection_color, thickness=2)

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec,
            connection_drawing_spec
        )

        # Collect landmark positions
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        # Ensure feature size matches model input (42 features for 1 hand)
        expected_features = 42
        actual_features = len(data_aux)

        if actual_features < expected_features:
            data_aux.extend([0] * (expected_features - actual_features))
        elif actual_features > expected_features:
            data_aux = data_aux[:expected_features]

        # Predict hand sign
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict.get(int(prediction[0]), "?")

        # Bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Display prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Sign Language Translator', frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
