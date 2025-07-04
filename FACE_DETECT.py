import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre_trained face mask detection model 
Face_Mask_DT_Model = load_model('Face_Mask_DT_Model.h5')



video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # black and white conversion
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        face_input = cv2.resize(face, (224, 224))
        face_input = face_input / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        prediction = Face_Mask_DT_Model.predict(face_input)[0]
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx]

        label = "Mask Detected" if class_idx == 0 else "No Mask"
        color = (0, 255, 0) if class_idx == 0 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break



video.release()
cv2.destroyAllWindows()