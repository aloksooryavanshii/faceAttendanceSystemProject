import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import *
from tkinter import messagebox

# ---------- IMAGE LOADING ----------
path = 'images'
images = []
classnames = []

if not os.path.exists(path):
    os.makedirs(path)

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        continue
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])

# ---------- ENCODE FUNCTION ----------
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_encodings(img)
        if faces:
            encodeList.append(faces[0])
    return encodeList

encodeListKnown = findEncodings(images)

# ---------- ATTENDANCE WRITING ----------
def markAttendance(name):
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("Name,Date,Time\n")

    with open("attendance.csv", "r+") as f:
        data = f.readlines()
        names = [line.split(',')[0] for line in data]

        if name not in names:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d,%H:%M:%S')
            f.writelines(f"\n{name},{dtString}")


# ---------- FACE RECOGNITION ----------
def startRecognition():
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Info", "Camera started. Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            messagebox.showerror("Error", "Camera not found!")
            break

        imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classnames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                markAttendance(name)

        cv2.imshow('Face Attendance System', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Session Ended!")


# ---------- TKINTER GUI ----------
root = Tk()
root.title("Face Recognition Attendance System")
root.geometry("500x300")
root.config(bg="#2C3E50")

Label(root, text="Face Recognition Attendance",
      font=("Helvetica", 18, "bold"), bg="#2C3E50", fg="white").pack(pady=30)

Button(root, text="Start Attendance", font=("Helvetica", 14, "bold"),
       bg="#27AE60", fg="white", command=startRecognition).pack(pady=20)

Button(root, text="Exit", font=("Helvetica", 14, "bold"),
       bg="#E74C3C", fg="white", command=root.destroy).pack(pady=10)

root.mainloop()

