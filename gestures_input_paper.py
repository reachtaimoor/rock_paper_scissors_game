import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Create directories for saving images
##gesture_directories = ['rock_images', 'paper_images', 'scissors_images']
gesture_directories = ['paper_images']
for gesture in gesture_directories:
    if not os.path.exists(gesture):
        os.makedirs(gesture)

# Function to clear previous images
def clear_previous_images():
    for gesture in gesture_directories:
        if os.path.exists(gesture):
            shutil.rmtree(gesture)  # Remove the directory and its contents
        os.makedirs(gesture)  # Recreate the directory

# Function to save image
def save_image(gesture_name, frame):
    image_count = len(os.listdir(gesture_name)) + 1
    cv2.imwrite(os.path.join(gesture_name, f'{gesture_name.split("_")[0]}_{image_count}.png'), frame)

# Function to capture images for a specific gesture
def capture_images_for_gesture(gesture_name, num_images=100):
    current_gesture = gesture_name
    gesture_prompt = f"Show your {current_gesture.split('_')[0].capitalize()} gesture. Press Space to capture images."

    images_captured = 0

    while images_captured < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(rgb_frame)

        # Create a mask for the hand
        mask = np.zeros(frame.shape, dtype=np.uint8)

        # Draw hand landmarks and create a mask
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the original frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Create a polygon for the hand area
                h, w, _ = frame.shape
                points = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
                points = np.array(points, np.int32)

                # Fill the mask with the hand area
                cv2.fillConvexPoly(mask, points, (255, 255, 255))

        # Dilation to increase the segmentation area
        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Create a black background
        black_background = np.zeros_like(frame)

        # Apply the dilated mask to the black background
        masked_frame = cv2.bitwise_and(black_background, dilated_mask)
        masked_frame += cv2.bitwise_and(frame, dilated_mask)

        # Display the gesture prompt on the frame
        cv2.putText(masked_frame, gesture_prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the masked frame
        cv2.imshow('Rock-Paper-Scissors', masked_frame)

        # Wait for spacebar to save the image
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar
            save_image(current_gesture, masked_frame)
            images_captured += 1
            print(f"{current_gesture.split('_')[0].capitalize()} image {images_captured}/{num_images} saved.")

    print(f"Captured {num_images} images for {current_gesture.split('_')[0].capitalize()}.")

# Clear previous images
clear_previous_images()

# Capture images for each gesture
#capture_images_for_gesture('rock_images')
capture_images_for_gesture('paper_images')
#capture_images_for_gesture('scissors_images')

cap.release()
cv2.destroyAllWindows()