import cv2

from yolo_for_tracking import *

# Initiliaze the YOLOv3 object detector.
detector = YOLO()

# Video capture device for the webcam.
video_capture = cv2.VideoCapture(0)

# Run inference on each frame from the webcam.
while True:

    # Read the next frame from the webcam.
    _, frame = video_capture.read()

    # Calculate the 50 percent of original dimensions.
    width = int(frame.shape[1] * 50 / 100)
    height = int(frame.shape[0] * 50 / 100)

    # Resize the image.
    new_frame = cv2.resize(frame, (width, height))

    # Run YOLOv3 inference.
    result, boxes = detector.inference(new_frame)

    # Display the frame with bounding boxes.
    cv2.imshow("Live Webcam", result)

    # Break the loop after pressing "x" key.
    if cv2.waitKey(1) &0XFF == ord('x'):

        break

# Close the capture device.
video_capture.release()
