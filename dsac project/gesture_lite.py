import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ------------------------------------------
# System control actions
# ------------------------------------------

def open_notifications():
    pyautogui.hotkey("win", "a")

def minimize_all():
    pyautogui.hotkey("win", "d")

def next_app():
    pyautogui.hotkey("alt", "tab")

def previous_app():
    pyautogui.hotkey("alt", "shift", "tab")

def volume_up():
    pyautogui.press("volumeup")

def volume_down():
    pyautogui.press("volumedown")


# ------------------------------------------
# Hand gesture helpers
# ------------------------------------------

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def is_pinch(lm):
    thumb = np.array(lm[4][:2])
    index = np.array(lm[8][:2])
    return dist(thumb, index) < 0.05


# ------------------------------------------
# Swipe detection
# ------------------------------------------

history = deque(maxlen=6)

def detect_swipe(center):
    history.append(center)

    if len(history) < 6:
        return None

    delta = history[-1] - history[0]

    # Vertical swipe
    if abs(delta[1]) > 0.15 and abs(delta[1]) > abs(delta[0]):
        return "UP" if delta[1] < 0 else "DOWN"

    # Horizontal swipe
    if abs(delta[0]) > 0.15 and abs(delta[0]) > abs(delta[1]):
        return "RIGHT" if delta[0] > 0 else "LEFT"

    return None


# ------------------------------------------
# Pinch Slide Volume Control (FIXED)
# ------------------------------------------

pinch_history = deque(maxlen=6)

def detect_pinch_slide(lm):
    """Detect pinch + vertical finger sliding movement."""
    if not is_pinch(lm):
        pinch_history.clear()
        return None

    index_y = lm[8][1]  # track index fingertip vertical movement
    pinch_history.append(index_y)

    if len(pinch_history) < 6:
        return None

    movement = pinch_history[-1] - pinch_history[0]

    if movement < -0.05:
        return "PINCH_UP"
    elif movement > 0.05:
        return "PINCH_DOWN"

    return None


# ------------------------------------------
# Cooldown controller
# ------------------------------------------

last_action = 0
cooldown = 0.5

def ready():
    global last_action
    if time.time() - last_action > cooldown:
        last_action = time.time()
        return True
    return False


# ------------------------------------------
# MAIN PROGRAM
# ------------------------------------------

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = [[lm.x, lm.y, lm.z] for lm in hand.landmark]

            # Track centroid for swipe detection
            cx = np.array([lm[9][0], lm[9][1]])

            # Detect swipe
            swipe = detect_swipe(cx)

            if swipe and not is_pinch(lm) and ready():
                if swipe == "UP":
                    open_notifications()
                elif swipe == "DOWN":
                    minimize_all()
                elif swipe == "LEFT":
                    previous_app()
                elif swipe == "RIGHT":
                    next_app()

                cv2.putText(frame, f"SWIPE {swipe}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

            # Detect pinch slide
            pinch_slide = detect_pinch_slide(lm)

            if pinch_slide == "PINCH_UP" and ready():
                volume_up()

            elif pinch_slide == "PINCH_DOWN" and ready():
                volume_down()

            if pinch_slide:
                cv2.putText(frame, f"{pinch_slide}", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

        cv2.putText(frame, "Swipe or Pinch-Slide", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        cv2.imshow("Gesture Control System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
