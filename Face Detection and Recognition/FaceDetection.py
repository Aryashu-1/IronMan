import cv2

cam = cv2.VideoCapture(0)
cascade_path = 'C:/Users/DELL/IronMan/Face Detection and Recognition/haarcascade_frontalcatface.xml'
# Load the Haar cascade
faceDetector = cv2.CascadeClassifier(cascade_path)
while True:
    success,bgr_frame = cam.read()
    if not success :
        continue
    
    all_faces = faceDetector.detectMultiScale(bgr_frame,1.3,3)
    print(all_faces)
    for face in all_faces:
        x,y,w,h = face
        cv2.rectangle(bgr_frame,(x,y),(x+w,y+h),(0,0,0),2)
    cv2.imshow("freme",bgr_frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
