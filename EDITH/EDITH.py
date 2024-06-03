import os
import cv2
import numpy as np
from collections import Counter
import tkinter as tk
from gtts import gTTS
import pygame
from io import BytesIO


#text to sppech converter
def text_to_speech(text):
    text = text  # Get text from the text entry widget
    if text:
        tts = gTTS(text=text, lang='en-au')  # For Australian English accent

        with BytesIO() as f:
            tts.write_to_fp(f)
            f.seek(0)
            pygame.mixer.init()
            pygame.mixer.music.load(f)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue


# Data
datasetPath = 'C:/Users/DELL/IronMan/Face Detection and Recognition/data/'
faceData = []
labels = []
labelMap = {}
classId = 0

for f in os.listdir(datasetPath):
    if f.endswith(".npy"):
        # X-values
        dataItem = np.load(os.path.join(datasetPath, f))
        print(dataItem.shape)

        faceData.append(dataItem)

        # Y-values
        m = dataItem.shape[0]
        target = classId * np.ones((m,))
        labels.append(target)
        labelMap[classId] = f[:-4]
        classId += 1

print(faceData)
print(labels)
print(labelMap)

X = np.concatenate(faceData, axis=0)
Y = np.concatenate(labels, axis=0).reshape((-1, 1))

print(X.shape)
print(Y.shape)

# Algorithm
def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def knn(X, y, xt, k=5):
    m = X.shape[0]
    dList = []
    for i in range(m):
        test_dist = dist(X[i], xt)
        dList.append((test_dist, y[i]))
    sorted_dList = sorted(dList)
    final_classes = [tuple(i[1]) for i in sorted_dList[:k]]
    final_classes_count = Counter(final_classes)
    result_class = final_classes_count.most_common(1)[0][0]
    return int(result_class[0])

# Taking input image for recognition
cascade_path = 'C:/Users/DELL/IronMan/Face Detection and Recognition/haarcascade_frontalface_alt.xml'
offset = 30
cam = cv2.VideoCapture(0)
faceDetector = cv2.CascadeClassifier(cascade_path)

if faceDetector.empty():
    raise IOError(f"Failed to load Haar cascade file from path: {cascade_path}")

while True:
    success, bgr_frame = cam.read()
    if not success:
        print("Reading camera failed")
        continue

    all_faces = faceDetector.detectMultiScale(bgr_frame, 1.3, 5)
    for face in all_faces:
        x, y, w, h = face
        
        # Check if the face region is valid
        if w <= 0 or h <= 0:
            print("Invalid face region detected")
            continue
        
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cropped_face = bgr_frame[y - offset:y + h + offset, x - offset:x + offset + w]
        
        # Check if the cropped face region is valid
        if cropped_face.size == 0:
            print("Cropped face region is empty")
            continue
        
        # Resize the cropped face
        cropped_face = cv2.resize(cropped_face, (100, 100))
        
        cropped_face = cropped_face.flatten().reshape(1, -1)
        print(cropped_face.shape)
        final_name_label = knn(X, Y, cropped_face, k=5)
        final_name = labelMap[final_name_label]
        print(final_name)
        text_to_speech(final_name)
        
        cv2.putText(bgr_frame, final_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Display the frame with face recognition results
    cv2.imshow("frame", bgr_frame)

    # Check for key press to exit the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


# Release camera and destroy all windows
cam.release()
cv2.destroyAllWindows()
