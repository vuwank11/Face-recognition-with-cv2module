# pip install face_recognition
# pip install cmake

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces.
bhuwan_img = face_recognition.load_image_file("Faces/BhuwanKhatiwada.JPG")
bhuwan_encoding = face_recognition.face_encodings(bhuwan_img)[0]

sujan_img = face_recognition.load_image_file("Faces/SujanKhatiwada.JPG")
sujan_encoding = face_recognition.face_encodings(sujan_img)[0]

known_faces_encodings = [bhuwan_encoding, sujan_encoding]
known_faces_names = ["Bhuwan Khatiwada", "Sujan Khatiwada"]


# list of expected people
people = known_faces_names.copy()

face_locations = []
face_encodings = []

# Get Current Date Time.
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize Faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        # Adding text to video.
        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.0
            fontColor = (0, 255, 0)
            thickness = 3
            lineType=2
            cv2.putText(frame, name + " in frame", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if name in people:
                people.remove(name)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Person", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
