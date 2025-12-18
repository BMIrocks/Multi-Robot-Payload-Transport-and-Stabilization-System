import cv2
import numpy as np

# ------------------------------
# CONFIGURATION
# ------------------------------
ROI = [0.5, 0.1, 0.95, 0.9]  
# ymin, xmin, ymax, xmax (relative)

THRESH_EDGE = 60     # Increase if too sensitive
MIN_AREA = 300       # Minimum contour area to be considered detection

# ------------------------------
# MAIN LOOP
# ------------------------------

cap = cv2.VideoCapture(0)  # 0 = laptop webcam

if not cap.isOpened():
    print("Could not access webcam!")
    exit()

print("Starting detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]

    # ------------------------------
    # 1. Extract Ground Region (ROI)
    # ------------------------------
    ymin = int(ROI[0] * h)
    xmin = int(ROI[1] * w)
    ymax = int(ROI[2] * h)
    xmax = int(ROI[3] * w)

    roi = frame[ymin:ymax, xmin:xmax]

    # ------------------------------
    # 2. Edge Detection
    # ------------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    sobel = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)  # vertical gradients
    abs_sobel = np.absolute(sobel)
    norm = (abs_sobel / (abs_sobel.max() + 1e-6)) * 255
    norm = norm.astype(np.uint8)

    # ------------------------------
    # 3. Threshold → Contour finding
    # ------------------------------
    _, th = cv2.threshold(norm, THRESH_EDGE, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ------------------------------
    # 4. Draw detections
    # ------------------------------
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        x, y, cw, ch = cv2.boundingRect(c)

        # Draw bounding box on original frame
        cv2.rectangle(frame, (xmin + x, ymin + y), (xmin + x + cw, ymin + y + ch),
                      (0, 255, 0), 2)

        cv2.putText(frame, "DETECTED OBJECT",
                    (xmin + x, ymin + y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Draw ROI box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    # ------------------------------
    # Show windows
    # ------------------------------
    cv2.imshow("Laptop Detector", frame)
    cv2.imshow("Processed ROI", th)

    # Quit safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
