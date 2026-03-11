import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

# các cặp điểm cần nối (skeleton của bàn tay)
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                points = []

                # vẽ các điểm landmark
                for lm in hand:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])

                    points.append((x,y))

                    cv2.circle(frame,(x,y),4,(0,255,0),-1)

                # vẽ các đường nối
                for c in connections:
                    pt1 = points[c[0]]
                    pt2 = points[c[1]]
                    cv2.line(frame, pt1, pt2, (255,0,0), 2)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()