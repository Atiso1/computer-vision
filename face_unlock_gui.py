import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time

# ================= LOAD MODEL =================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/model.yml")

label_dict = np.load("trainer/labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ================= SETTINGS =================
confidence_threshold = 90
unlock_threshold = 2

blink_counter = 0
blink_required = 1
eyes_last = True

unlock_counter = 0
locked = True

last_face_time = time.time()
timeout_lock = 2


# ================= GUI =================
root = tk.Tk()
root.title("AI Face Unlock System")
root.geometry("900x650")
root.configure(bg="#1e1e1e")

title = tk.Label(root,
                 text="AI FACE UNLOCK SYSTEM",
                 font=("Arial",24,"bold"),
                 bg="#1e1e1e",
                 fg="white")
title.pack(pady=10)

status_label = tk.Label(root,
                        text="LOCKED",
                        font=("Arial",28,"bold"),
                        bg="#1e1e1e",
                        fg="red")
status_label.pack()

info_label = tk.Label(root,
                      text="Blink to verify",
                      font=("Arial",14),
                      bg="#1e1e1e",
                      fg="white")
info_label.pack()

video_label = tk.Label(root)
video_label.pack(pady=10)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
running = False


def start_camera():
    global running
    running = True
    update_frame()


def stop_camera():
    global running
    running = False


def update_frame():

    global blink_counter
    global eyes_last
    global unlock_counter
    global locked
    global last_face_time

    if not running:
        return

    ret, frame = cap.read()

    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces) == 0:

        unlock_counter = 0

        if time.time() - last_face_time > timeout_lock:
            locked = True

    for (x,y,w,h) in faces:

        last_face_time = time.time()

        face_roi = gray[y:y+h,x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi)

        if len(eyes)==0 and eyes_last:
            blink_counter+=1
            eyes_last=False
        elif len(eyes)>0:
            eyes_last=True

        if blink_counter>=blink_required:

            face_roi=cv2.resize(face_roi,(200,200))

            id_,confidence=recognizer.predict(face_roi)

            if confidence<confidence_threshold:

                name=label_dict[id_]
                unlock_counter+=1
                color=(0,255,0)

            else:

                name="Unknown"
                unlock_counter=0
                color=(0,0,255)

            if unlock_counter>=unlock_threshold:
                locked=False

            cv2.putText(frame,name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,color,2)

        else:

            cv2.putText(frame,"Blink to verify",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(0,0,255),2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # ===== UPDATE GUI STATUS =====

    if locked:
        status_label.config(text="LOCKED",fg="red")
    else:
        status_label.config(text="UNLOCKED",fg="lime")

    info_label.config(text=f"Blinks: {blink_counter}")

    # ===== SHOW FRAME =====

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10,update_frame)


# ================= BUTTONS =================

btn_frame = tk.Frame(root,bg="#1e1e1e")
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame,
                      text="Start Camera",
                      font=("Arial",12),
                      command=start_camera,
                      bg="#4CAF50",
                      fg="white",
                      width=15)

start_btn.grid(row=0,column=0,padx=10)

stop_btn = tk.Button(btn_frame,
                     text="Stop Camera",
                     font=("Arial",12),
                     command=stop_camera,
                     bg="#f44336",
                     fg="white",
                     width=15)

stop_btn.grid(row=0,column=1,padx=10)

root.mainloop()

cap.release()
cv2.destroyAllWindows()