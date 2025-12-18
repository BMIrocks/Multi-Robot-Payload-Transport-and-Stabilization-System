"""
Advanced Hand Gesture Controller
==================================

SETUP:
------
pip install opencv-python mediapipe numpy pyautogui

USAGE:
------
- Show BOTH hands to the camera
- LEFT HAND = Mode Selection:
  * FIST = Mouse Mode
  * OPEN = Scroll Mode
  * PINCH = Zoom Mode
  * PEACE = Navigation Mode (default)

- RIGHT HAND = Perform Actions (based on current mode)

CONTROLS BY MODE:
-----------------
NAVIGATION Mode (Peace sign on left):
  - Swipe UP = Open notifications
  - Swipe DOWN = Minimize all windows
  - Swipe LEFT = Previous app
  - Swipe RIGHT = Next app

MOUSE Mode (Fist on left):
  - Move hand = Move cursor
  - PINCH = Left click
  - FIST = Right click

SCROLL Mode (Open hand on left):
  - Move hand UP/DOWN = Scroll

ZOOM Mode (Pinch on left):
  - OPEN hand = Zoom in
  - FIST = Zoom out

VOLUME (Works in all modes):
  - Pinch thumb+index and slide up/down

Press 'q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque
import sys

# Safety settings
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# -------------------------------------------------------
# System Actions
# -------------------------------------------------------

def open_notifications():
    """Open system notification center"""
    try:
        pyautogui.hotkey("win", "a")
    except Exception as e:
        print(f"Error opening notifications: {e}")


def minimize_all():
    """Minimize all windows"""
    try:
        pyautogui.hotkey("win", "d")
    except Exception as e:
        print(f"Error minimizing: {e}")


def next_app():
    """Switch to next application"""
    try:
        pyautogui.hotkey("alt", "tab")
    except Exception as e:
        print(f"Error switching app: {e}")


def previous_app():
    """Switch to previous application"""
    try:
        pyautogui.hotkey("alt", "shift", "tab")
    except Exception as e:
        print(f"Error switching app: {e}")


def volume_up():
    """Increase system volume"""
    try:
        pyautogui.press("volumeup")
    except Exception as e:
        print(f"Error adjusting volume: {e}")


def volume_down():
    """Decrease system volume"""
    try:
        pyautogui.press("volumedown")
    except Exception as e:
        print(f"Error adjusting volume: {e}")


# -------------------------------------------------------
# Gesture Math Helpers
# -------------------------------------------------------

def dist(a, b):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(a) - np.array(b))


def curl_score(tip, pip):
    """Calculate how curled a finger is (0=extended, 1=curled)"""
    return max(0, min(1, (tip[1] - pip[1]) * 6))


def is_extended(tip, pip):
    """Check if finger is extended"""
    return tip[1] < pip[1]


def gesture_confidence(score):
    """Convert score to percentage"""
    return round(score * 100, 1)


# -------------------------------------------------------
# Swipe Detection
# -------------------------------------------------------
history = deque(maxlen=6)


def detect_swipe(center):
    """Detect swipe direction from hand movement"""
    history.append(center)
    if len(history) < 6:
        return None

    delta = history[-1] - history[0]

    # Vertical swipes
    if abs(delta[1]) > 0.15 and abs(delta[1]) > abs(delta[0]):
        return "UP" if delta[1] < 0 else "DOWN"

    # Horizontal swipes
    if abs(delta[0]) > 0.15 and abs(delta[0]) > abs(delta[1]):
        return "RIGHT" if delta[0] > 0 else "LEFT"

    return None


# -------------------------------------------------------
# Mouse & Interaction Actions
# -------------------------------------------------------

def mouse_move(x, y):
    """Move mouse cursor smoothly"""
    try:
        screen_w, screen_h = pyautogui.size()
        pyautogui.moveTo(int(x * screen_w), int(y * screen_h), duration=0)
    except Exception as e:
        print(f"Error moving mouse: {e}")


def left_click():
    """Perform left mouse click"""
    try:
        pyautogui.click()
    except Exception as e:
        print(f"Error clicking: {e}")


def right_click():
    """Perform right mouse click"""
    try:
        pyautogui.click(button="right")
    except Exception as e:
        print(f"Error right-clicking: {e}")


def scroll(amount):
    """Scroll by specified amount"""
    try:
        pyautogui.scroll(int(amount))
    except Exception as e:
        print(f"Error scrolling: {e}")


def zoom_in():
    """Zoom in (Ctrl+Plus)"""
    try:
        pyautogui.hotkey("ctrl", "+")
    except Exception as e:
        print(f"Error zooming: {e}")


def zoom_out():
    """Zoom out (Ctrl+Minus)"""
    try:
        pyautogui.hotkey("ctrl", "-")
    except Exception as e:
        print(f"Error zooming: {e}")


# -------------------------------------------------------
# Pinch Slide Volume Control
# -------------------------------------------------------
pinch_history = deque(maxlen=6)


def detect_pinch_slide(lm):
    """Detect pinch and slide gesture for volume control"""
    thumb, index = lm[4], lm[8]
    
    # Check if fingers are pinched
    if dist(thumb[:2], index[:2]) > 0.05:
        pinch_history.clear()
        return None

    pinch_history.append(index[1])
    if len(pinch_history) < 6:
        return None

    movement = pinch_history[-1] - pinch_history[0]

    if movement < -0.04:
        return "PINCH_UP"
    elif movement > 0.04:
        return "PINCH_DOWN"

    return None


# -------------------------------------------------------
# Hand Gesture Classification
# -------------------------------------------------------

def classify_basic(lm):
    """Classify hand gesture with confidence score"""
    index_tip, index_pip = lm[8], lm[6]
    mid_tip, mid_pip = lm[12], lm[10]
    ring_tip, ring_pip = lm[16], lm[14]
    pink_tip, pink_pip = lm[20], lm[18]
    thumb_tip, thumb_ip = lm[4], lm[3]

    # FIST score (all fingers curled)
    f = (curl_score(index_tip, index_pip) +
         curl_score(mid_tip, mid_pip) +
         curl_score(ring_tip, ring_pip) +
         curl_score(pink_tip, pink_pip)) / 4

    # OPEN score (opposite of fist)
    o = 1 - f

    # PINCH score (thumb and index close)
    p = max(0, min(1, (0.08 - dist(thumb_tip[:2], index_tip[:2])) * 15))

    # PEACE score (index and middle extended, others curled)
    peace = (is_extended(index_tip, index_pip) *
             is_extended(mid_tip, mid_pip) *
             (1 - curl_score(ring_tip, ring_pip)) *
             (1 - curl_score(pink_tip, pink_pip)))

    gestures = {"FIST": f, "OPEN": o, "PINCH": p, "PEACE": peace}
    gesture = max(gestures, key=gestures.get)
    return gesture, gestures[gesture]


# -------------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------------

def main():
    """Main gesture control loop"""
    
    # Check camera availability
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot access camera!")
        print("Please check:")
        print("1. Camera is connected")
        print("2. No other application is using the camera")
        print("3. Camera permissions are granted")
        sys.exit(1)
    
    print("Camera initialized successfully!")
    print(__doc__)
    
    mode = "NAVIGATION"
    last_action = 0
    cooldown = 0.5
    
    # Action cooldowns for different modes
    last_click = 0
    click_cooldown = 0.3
    last_scroll = 0
    scroll_cooldown = 0.1
    last_zoom = 0
    zoom_cooldown = 0.3

    try:
        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    num_hands = len(result.multi_hand_landmarks)
                    
                    for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                        mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS
                        )

                        lm = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        gesture, score = classify_basic(lm)
                        cx = np.array([lm[9][0], lm[9][1]])

                        # LEFT HAND = MODE SELECTOR (First detected hand)
                        if i == 0:
                            if gesture == "FIST" and score > 0.6:
                                mode = "MOUSE"
                            elif gesture == "OPEN" and score > 0.6:
                                mode = "SCROLL"
                            elif gesture == "PINCH" and score > 0.7:
                                mode = "ZOOM"
                            elif gesture == "PEACE" and score > 0.6:
                                mode = "NAVIGATION"

                        # RIGHT HAND = PERFORM ACTION (Second detected hand)
                        elif i == 1:
                            conf = gesture_confidence(score)

                            # NAVIGATION MODE
                            if mode == "NAVIGATION":
                                swipe = detect_swipe(cx)
                                if swipe and conf > 60 and time.time() - last_action > cooldown:
                                    last_action = time.time()
                                    if swipe == "UP":
                                        open_notifications()
                                    elif swipe == "DOWN":
                                        minimize_all()
                                    elif swipe == "LEFT":
                                        previous_app()
                                    elif swipe == "RIGHT":
                                        next_app()

                            # MOUSE MODE
                            elif mode == "MOUSE":
                                mouse_move(lm[9][0], lm[9][1])
                                
                                if gesture == "PINCH" and conf > 70 and time.time() - last_click > click_cooldown:
                                    left_click()
                                    last_click = time.time()
                                elif gesture == "FIST" and conf > 70 and time.time() - last_click > click_cooldown:
                                    right_click()
                                    last_click = time.time()

                            # SCROLL MODE
                            elif mode == "SCROLL":
                                if gesture == "OPEN" and conf > 60 and time.time() - last_scroll > scroll_cooldown:
                                    scroll_amount = int((lm[9][1] - 0.5) * -80)
                                    scroll(scroll_amount)
                                    last_scroll = time.time()

                            # ZOOM MODE
                            elif mode == "ZOOM":
                                if time.time() - last_zoom > zoom_cooldown:
                                    if gesture == "OPEN" and conf > 60:
                                        zoom_in()
                                        last_zoom = time.time()
                                    elif gesture == "FIST" and conf > 60:
                                        zoom_out()
                                        last_zoom = time.time()

                            # VOLUME CONTROL (Works in all modes)
                            vol = detect_pinch_slide(lm)
                            if vol and time.time() - last_action > cooldown:
                                if vol == "PINCH_UP":
                                    volume_up()
                                    last_action = time.time()
                                elif vol == "PINCH_DOWN":
                                    volume_down()
                                    last_action = time.time()

                        # Display gesture info
                        cv2.putText(
                            frame, 
                            f"{'LEFT' if i == 0 else 'RIGHT'}: {gesture} ({conf:.1f}%)",
                            (10, 40 + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8,
                            (0, 255, 0), 
                            2
                        )

                # Display current mode
                cv2.putText(
                    frame, 
                    f"MODE: {mode}", 
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0,
                    (0, 255, 255), 
                    2
                )
                
                # Instructions
                cv2.putText(
                    frame, 
                    "Press 'q' to quit", 
                    (w - 250, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6,
                    (255, 255, 255), 
                    1
                )

                cv2.imshow("Hand Gesture Controller", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete!")


if __name__ == "__main__":
    main()