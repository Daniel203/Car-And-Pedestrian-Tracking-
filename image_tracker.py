import cv2

IMG_FILE = "assets/car_image.jpg"
CLASSIFIER_FILE = "car_detector.xml"

# create opencv image
img = cv2.imread(IMG_FILE)

# convert to grayscale
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(CLASSIFIER_FILE)

# detect cars
cars = car_tracker.detectMultiScale(black_and_white)

# draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


cv2.imshow("Car and Pedestrian Detector", img)
cv2.waitKey()