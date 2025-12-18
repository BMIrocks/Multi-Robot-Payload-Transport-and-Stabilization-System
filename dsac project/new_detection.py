import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------
# Load YOLO model (COCO classes: includes cell phone, laptop)
# ----------------------------------------------------------
yolo_model = YOLO("yolov8n.pt")   # auto-downloads on first run

# ----------------------------------------------------------
# MediaPipe Hands
# ----------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# ----------------------------------------------------------
# Gesture rules (same as original, expanded)
# ----------------------------------------------------------
def detect_gesture(lm):
    thumb_tip = lm[4]
    index_tip = lm[8]
    middle_tip = lm[12]
    ring_tip = lm[16]
    pinky_tip = lm[20]

    if dist(thumb_tip, index_tip) < 0.05:
        return "PINCH 🤏"

    if (lm[8][1] > lm[6][1] and lm[12][1] > lm[10][1] and 
        lm[16][1] > lm[14][1] and lm[20][1] > lm[18][1]):
        return "FIST ✊"

    if (lm[8][1] < lm[6][1] and lm[12][1] > lm[10][1] and 
        lm[16][1] > lm[14][1] and lm[20][1] > lm[18][1]):
        return "POINT 👉"

    if (lm[4][1] < lm[3][1] and lm[8][1] > lm[6][1] and 
        lm[12][1] > lm[10][1] and lm[16][1] > lm[14][1] and lm[20][1] > lm[18][1]):
        return "THUMBS UP 👍"

    if (lm[4][1] > lm[3][1] and lm[8][1] > lm[6][1] and 
        lm[12][1] > lm[10][1] and lm[16][1] > lm[14][1] and lm[20][1] > lm[18][1]):
        return "THUMBS DOWN 👎"

    if (lm[8][1] < lm[6][1] and lm[12][1] < lm[10][1] and 
        lm[16][1] > lm[14][1] and lm[20][1] > lm[18][1]):
        return "PEACE ✌️"

    extended = sum([
        lm[8][1] < lm[6][1],
        lm[12][1] < lm[10][1],
        lm[16][1] < lm[14][1],
        lm[20][1] < lm[18][1]
    ])

    if extended >= 3:
        return "OPEN HAND ✋"

    return "UNKNOWN"

# ----------------------------------------------------------
# Camera + Hand detection
# ----------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------
        # YOLO Object Detection (phones, faces, tablets, etc.)
        # --------------------------------------------------
        results = yolo_model(rgb, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]  # class name
                conf  = float(box.conf[0])

                # Filter only what you want:
                if label in ["cell phone", "laptop", "tv", "person"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    color = (0, 255, 255) if label == "cell phone" else \
                            (255, 0, 255) if label == "laptop" else \
                            (0, 255, 0)   # generic

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)

        # --------------------------------------------------
        # Hand Gesture Detection
        # --------------------------------------------------
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                lm = [[lm.x, lm.y, lm.z] for lm in hand.landmark]
                gesture = detect_gesture(lm)

                cv2.putText(frame, gesture, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 0), 3)

        cv2.imshow("Gestures + YOLO Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
