import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
pyautogui.FAILSAFE=False
# Set up mediapipe hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Get system screen size
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Function to convert MediaPipe normalized coordinates to screen coordinates
def normalized_to_pixel_coordinates(normalized_x, normalized_y):
    x_px = min(int(normalized_x * SCREEN_WIDTH), SCREEN_WIDTH - 1)
    y_px = min(int(normalized_y * SCREEN_HEIGHT), SCREEN_HEIGHT - 1)
    return x_px, y_px

# Open webcam
cap = cv2.VideoCapture(0)
click_state=False
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the frame horizontally to remove the mirroring effect
    image = cv2.flip(image, 1)

    # Resize the frame to the system screen size
    image = cv2.resize(image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image_rgb

    # Process the image with MediaPipe
    results = hands.process(image)
    i=0
    xcord = []
    ycord = []
    xdiff=0
    ydiff=0
    # Extract hand landmarks if available
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            # Get coordinates of index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = normalized_to_pixel_coordinates(index_finger_tip.x, index_finger_tip.y)
            xcord.append(x)
            ycord.append(y)
            i+=1
            if i > 1 :
                xdiff = xcord[i]-xcord[i-1]
                ydiff = ycord[i]-ycord[i-1]
                xcord=[]
                ycord=[]
                i=0
            if xdiff >0:
                x = 1.15*x

            else:
                x= x/1.15
            if ydiff > 0:
                y*=1.15
            else:
                y/=1.15
            # Move the mouse pointer
            pyautogui.moveTo(x,y)
            # Calculate differences from previous coordinates
            tx1,ty1 = hand_landmarks.landmark[3].x,hand_landmarks.landmark[3].y
            px2,py2 = hand_landmarks.landmark[5].x,hand_landmarks.landmark[5].y
            xtdiff = abs(tx1-px2)
            ytdiff = abs(ty1-py2)
            xtdiff= int(xtdiff*100)
            ytdiff=int(ytdiff*100)
            print(xtdiff,ytdiff)
            # difft = (((xdiff**2) + (ydiff**2) )**0.5) 
            # print(difft)
            # cv2.circle(image, (tx1, ty1), 5, (255, 0, 255), cv2.FILLED)
            # cv2.circle(image, (px2, py2), 5, (255, 0, 255), cv2.FILLED)
            # Check for click gesture
            if  ytdiff<=5:
                if not click_state:
                    #Perform click action
                    pyautogui.click()
                    click_state = True
            else:
                click_state = False

    # Display the image with landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit by pressing Esc key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
