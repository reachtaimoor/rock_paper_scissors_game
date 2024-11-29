import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('rps_model.h5')  # Ensure you have a trained model saved as 'gesture_model.h5'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define a mapping for gesture predictions
gesture_labels = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Unable to read from camera.")
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(rgb_frame)

    # Create a mask for the hand
    mask = np.zeros(frame.shape, dtype=np.uint8)

    # Initialize gesture variables
    gesture = "Unknown"
    confidence = 0.0

    # Draw hand landmarks and create a mask
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the original frame (for visualization, can be omitted)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

    # If hand landmarks are detected, predict the gesture
    if results.multi_hand_landmarks:
        # Crop the hand region from the frame
        hand_image = frame[0:frame.shape[0], 0:frame.shape[1]]  # Use the full frame for prediction
        hand_image_resized = cv2.resize(hand_image, (300, 300))
        hand_image_normalized = hand_image_resized / 255.0  # Normalize to [0, 1]
        hand_image_input = np.expand_dims(hand_image_normalized, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(hand_image_input)
        predicted_class = np.argmax(prediction, axis=1)[0]
        gesture = gesture_labels.get(predicted_class, "Unknown")

        # Get confidence
        confidence = np.max(prediction)

    # Display the predicted gesture and confidence on the masked frame
    cv2.putText(masked_frame, f'Gesture: {gesture}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(masked_frame, f'Confidence: {confidence:.2f}', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the output frame
    cv2.imshow('Segmented Hand with Prediction', masked_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()