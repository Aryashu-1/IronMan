import cv2
import pyautogui

# Initialize OpenCV's face detector (we will use a pre-trained Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Set up video capture
cap = cv2.VideoCapture(0)

# Loop to capture frames
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) containing the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Iterate over detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Calculate the center of the eye
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2

            # Move the mouse pointer based on the eye position (simplified example)
            pyautogui.moveTo(eye_center_x, eye_center_y)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
