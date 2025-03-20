import cv2
import mediapipe as mp

# Initialize MediaPipe for Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Set up the hand tracking model with confidence thresholds
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the webcam rree
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam is open.")

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame from BGR (OpenCV default) to RGB (required for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)  # Process frame with MediaPipe model

    # If hands are detected, draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates and label them on the screen
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape  # Get frame dimensions
                cx, cy = int(landmark.x * w), int(landmark.y * h)  # Convert to pixel values

                # Display landmark index at detected points
                cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame with detected hand landmarks
    cv2.imshow("Hand Tracking - Press 'q' to Quit", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

# Release resources and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
