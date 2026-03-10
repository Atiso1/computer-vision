import cv2
import os
import numpy as np

def train_model():
    dataset_path = "dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    if not os.path.exists(dataset_path):
        print("[ERROR] Dataset not found!")
        return

    for user in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user)
        if not os.path.isdir(user_path):
            continue

        label_dict[current_label] = user

        for image_name in os.listdir(user_path):
            img_path = os.path.join(user_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        print("[ERROR] No training data found!")
        return

    os.makedirs("trainer", exist_ok=True)

    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer/model.yml")
    np.save("trainer/labels.npy", label_dict)

    print(f"[INFO] Training completed with {len(faces)} images.")

if __name__ == "__main__":
    train_model()