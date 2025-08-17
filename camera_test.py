import cv2
import os

# rtsp_url = "rtsp://admin1:admin123@103.86.197.115:8082/live"
rtsp_url = "rtsp://admin1:admin123@103.86.197.115:8082/live"
cap = cv2.VideoCapture(rtsp_url)
# cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)  # Force FFMPEG

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
# cap = cv2.VideoCapture("rtsp://admin1:admin123@103.86.197.115:8082", cv2.CAP_FFMPEG)


# my_cam = 'rtsp://admin:admin123@192.168.1.108/live'
# cap = cv2.VideoCapture(my_cam)

if not cap.isOpened():
    print("Failed to open RTSP stream. Check the URL or OpenCV installation.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    frame_resized = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("RTSP Stream", frame_resized)

    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
