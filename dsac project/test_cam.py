import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not available")
    exit()
ret, frame = cap.read()
print("Frame read:", ret, "Frame shape:", None if frame is None else frame.shape)
cap.release()
