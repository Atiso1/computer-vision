import cv2
import os

def collect_faces():
    user_name = input("Enter user name: ").strip()
    save_path = os.path.join("dataset", user_name)
    os.makedirs(save_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    count = 0
    max_samples = 100

    print("[INFO] Collecting face samples... Look at the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            count += 1
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Samples: {count}/{max_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

        cv2.imshow("Collecting Faces", frame)

        if cv2.waitKey(1) & 0xFF == 27 or count >= max_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Data collection completed!")

if __name__ == "__main__":
    collect_faces()