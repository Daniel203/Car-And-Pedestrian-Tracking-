import cv2

VIDEO_FILE = "assets/video2.mp4"
CLASSIFIER_FILE = "car_detector.xml"
PEDESTRIAN_FILE = "haarcascades_fullbody.xml"

# create opencv video
video = cv2.VideoCapture(VIDEO_FILE)

# create the classifier
car_tracker = cv2.CascadeClassifier(CLASSIFIER_FILE)
pedestrian_tracker = cv2.CascadeClassifier(PEDESTRIAN_FILE)

while True:
    # read current frame
    read_succesful, frame = video.read()

    if read_succesful:
        gray_scaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrian
    cars = car_tracker.detectMultiScale(gray_scaled)
    pedestrinas = pedestrian_tracker.detectMultiScale(gray_scaled)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    for (x, y, w, h) in pedestrinas:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Car and Pedestrian Detector", frame)
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key==81 or key==113:
        break

video.release()
