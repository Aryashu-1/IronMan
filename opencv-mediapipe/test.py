import cv2
import mediapipe as mp
import time


##creating object for open cv to use camera
cam = cv2.VideoCapture(0)
pTime =0
cTime =0
# mediapipe objects 
#drwas lines to connect the points on hands
mpDraw = mp.solutions.drawing_utils
#detects the hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
while True:
    ret,frame = cam.read()
    #image should be converted to rgb
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #process method to process hands
    results = hands.process(frameRGB)
    #results.multi_hand_landmarks gives all the coordinates of landmarks of hands and we draw points and connect them 
    if results.multi_hand_landmarks :
        for handLms in results.multi_hand_landmarks : 
            # #extracting all landmarks into list
            # for id,lm in enumerate(handLms.landmark) :
            #     h,w,c = frame.shape
            #     cx,cy,cz = int(lm.x*w), int(lm.y*h), int(lm.z)
            #     print(id,cx,cy,cz)
            h,w,c = frame.shape
            x1,y1,z1 = (handLms.landmark[12].x*w),(handLms.landmark[12].y*h), (handLms.landmark[12].z)
            x2,y2,z2 = (handLms.landmark[0].x*w),(handLms.landmark[0].y*h), (handLms.landmark[0].z)
            z = handLms.landmark[9].z * 1000
            distance = int((((x2 - x1)**2) + ((y2-y1)**2))**0.5)
            print(z,distance)

                
                # #draw circle around any particular point
                # if id == 9 :
                #     cv2.circle(frame,(cx,cy),15,(255,255,255),cv2.FILLED)

            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
    
    #calculating fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #displaying fps as text on frame
    cv2.putText(frame,str(int(fps)), (10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3) 


    if ret == False :
        continue
    #display frame
    cv2.imshow("My frame",frame)
    #terminating cam
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q') :
        break

cam.release()
cv2.destroyAllWindows 