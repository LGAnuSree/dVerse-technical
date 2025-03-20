import cv2

# Try different camera indices (0, 1, 2, etc.)
for i in range(3):  # Check first 3 indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available.")
        cap.release()
    else:
        print(f"Camera {i} is not available.")
