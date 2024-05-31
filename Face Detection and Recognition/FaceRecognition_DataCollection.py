import cv2
import numpy as np
#1--collect data by taking 20images of each person
cam = cv2.VideoCapture(0)
#1--1-taking name of the person as input and store the images into data folder
fileName = input("Enter Your Name")
dataset_path = 'C:/Users/DELL/IronMan/Face Detection and Recognition/data/'
cascade_path = 'C:/Users/DELL/IronMan/Face Detection and Recognition/haarcascade_frontalface_alt.xml'
offset =30
facedata = []
skip=0
cnt=0
# Load the Haar cascade
faceDetector = cv2.CascadeClassifier(cascade_path)


if faceDetector.empty():
    raise IOError(f"Failed to load Haar cascade file from path: {cascade_path}")
while True:
    success,bgr_frame = cam.read()
    if not success :
        print("reading Camera Failed")
        continue
    #converting to gray scale to reduce storage
    gray_frame = cv2.cvtColor(bgr_frame,cv2.COLOR_BGR2GRAY)

    if not success :
        print("reading Camera Failed")
        continue
    
    all_faces = faceDetector.detectMultiScale(bgr_frame,1.3,5)

    all_faces = sorted(all_faces,key= lambda face: face[2]*face[3])
    if len(all_faces)>0:
        final_faces=all_faces[-1]

        x,y,w,h = final_faces
        cv2.rectangle(bgr_frame,(x,y),(x+w,y+h),(0,0,0),2)
        cropped_face = bgr_frame[y-offset:y+h+offset,x-offset:x+offset+w]

        cropped_face = cv2.resize(cropped_face,(100,100))
        cv2.imshow("Cropped Face",cropped_face)
        skip+=1
        if skip % 10 == 0:
            cnt+=1
            facedata.append(cropped_face)
            print( "saved " + str(len(facedata)))
            
        if cnt == 20:
            break
        
    
    cv2.imshow("frame",bgr_frame)
   
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#Write data on the disk
facedata = np.asarray(facedata)
m = facedata.shape[0]
facedata=facedata.reshape((m,-1))

#save on disk as np array
filePath = dataset_path + fileName + ".npy"
np.save(filePath,facedata)
print("data saved successfully")

#release camera and destroy all windows
cam.release()
cv2.destroyAllWindows