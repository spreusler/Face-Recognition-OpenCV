import cv2

# Load pre-trained model 'haarcascade_frontalface_alt.xml" for face recognition / classification
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Create object for default camera 0
video_capture = cv2.VideoCapture(0)

while True:
    # Get frame from video
    ret, frame = video_capture.read()

    # Define grey for face box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Append rectangle to each face in a frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show frame as video stream
    cv2.imshow('Video', frame)

    # Run program until user enters 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# If program exists while loop, release video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()