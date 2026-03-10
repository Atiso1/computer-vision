import cv2
import numpy as np
import time

# ================= LOAD MODEL =================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/model.yml")

label_dict = np.load("trainer/labels.npy", allow_pickle=True).item()

# ================= CASCADE =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

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

print("Blink to verify liveness...")

# ================= LOOP =================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ================= NO FACE =================
    if len(faces) == 0:

        unlock_counter = 0

        if time.time() - last_face_time > timeout_lock:
            locked = True

    # ================= FACE DETECTED =================
    for (x, y, w, h) in faces:

        last_face_time = time.time()

        face_roi = gray[y:y+h, x:x+w]

        # ===== EYE DETECTION =====
        eyes = eye_cascade.detectMultiScale(face_roi)

        # ===== BLINK DETECTION =====
        if len(eyes) == 0 and eyes_last:
            blink_counter += 1
            eyes_last = False

        elif len(eyes) > 0:
            eyes_last = True

        cv2.putText(frame,
                    f"Blinks: {blink_counter}",
                    (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,0),
                    2)

        # ===== FACE RECOGNITION AFTER BLINK =====
        if blink_counter >= blink_required:

            face_roi = cv2.resize(face_roi, (200,200))

            id_, confidence = recognizer.predict(face_roi)

            print("confidence:", confidence)

            if confidence < confidence_threshold:

                name = label_dict[id_]
                unlock_counter += 1
                color = (0,255,0)

            else:

                name = "Unknown"
                unlock_counter = 0
                color = (0,0,255)

            if unlock_counter >= unlock_threshold:
                locked = False

            cv2.putText(frame,
                        f"{name} ({round(confidence,1)})",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2)

        else:

            cv2.putText(frame,
                        "Blink to verify",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,0,255),
                        2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # ================= STATUS =================
    status_text = "UNLOCKED" if not locked else "LOCKED"

    status_color = (0,255,0) if not locked else (0,0,255)

    cv2.putText(frame,
                status_text,
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                status_color,
                3)

    cv2.imshow("Face Unlock System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()