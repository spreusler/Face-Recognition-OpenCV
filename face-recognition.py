import cv2
import sys

# Laden des bereits trainierten Modells 'haarcascade_frontalface_alt.xml"
# zur Klassifikation / Erkennung von Gesichtern
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Erzeugung eines Objekts, das auf die Default Kamera (Webcam Parameter 0) zugreift
video_capture = cv2.VideoCapture(0)

while True:
    # Aufgreifen des Bildes pro Frame
    ret, frame = video_capture.read()

    # Zuweisung der Farbe grau fuer Box Umrandung des Gesichts
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Erkennung mehrerer Gesichter
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Erzeugen eines Rechtecks fuer jedes der erkannten Gesichter und anfuegen in Frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Ausgabe der Frames
    cv2.imshow('Video', frame)

    # Solange laufen bis 'q' gedrueckt laeuft das Programm. Ansonsten springt das
    # Programm mit 'break' aus while-Schleife
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sofern Programm aus while-Schleife springt, erfolgt die Freigabe des Objekts und Beendigung aller Fenster
video_capture.release()
cv2.destroyAllWindows()