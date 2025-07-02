import cv2
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # black and white conversion
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) # draw rectangle around face
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("WindowFrame", frame) # name of frame
    k = cv2.waitKey(1)
    if k == ord('q'):
        break



video.release()
cv2.destroyAllWindows()