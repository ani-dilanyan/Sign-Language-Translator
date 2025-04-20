import os
import cv2

# Directory where collected data will be stored
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of sign classes and dataset size per class
number_of_classes = 26 
dataset_size = 100  # Number of images per class

# Open webcam (0 is the default camera; try 1 or 2 if 0 doesn't work)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop through each class to collect images
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)  # Create directory for class if it doesn't exist

    print(f'Collecting data for class {j}')

    # Display message before capturing images
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue  # Skip if frame capture failed

        # Display an instruction message on the video feed
        cv2.putText(frame, 'Ready? Press "Q" to start capturing!', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Wait for the user to press 'q' to start capturing images
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue  # Skip iteration if frame capture failed

        # Show the frame while capturing images
        cv2.imshow('frame', frame)
        cv2.waitKey(10) 

        # Save the captured frame in the respective class folder
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()