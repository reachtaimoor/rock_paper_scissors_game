import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Function to determine gesture
def detect_gesture(landmarks):
    index_finger_tip = landmarks[8][1]
    index_finger_base = landmarks[6][1]
    middle_finger_tip = landmarks[12][1]
    middle_finger_base = landmarks[10][1]

    if index_finger_tip < index_finger_base and middle_finger_tip < middle_finger_base:
        return "Rock"
    elif index_finger_tip < index_finger_base and middle_finger_tip > middle_finger_base:
        return "Scissors"
    elif index_finger_tip > index_finger_base and middle_finger_tip < middle_finger_base:
        return "Paper"
    return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            gesture = detect_gesture(landmarks)

            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the gesture
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()