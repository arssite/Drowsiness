import cv2
import numpy as np
from keras.models import load_model
import time
import pygame

# Load the trained model
model = load_model('Driver_Drowsiness_Detection.h5')

# Initialize pygame for sound alert
pygame.mixer.init()

# Load the cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect drowsiness
def detect_drowsiness(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey + eh, ex:ex + ew]
            eye_roi_color = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB)  # Convert grayscale to color
            resized = cv2.resize(eye_roi_color, (32, 32))  # Resize the image to (32, 32)
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (-1, 32, 32, 3))  # Reshape to match model input shape
            result = model.predict(reshaped)
            print("Result",result[0][1])
            if result[0][1] > 0.01:
                return False

    return True


# Main function
def main():
    cap = cv2.VideoCapture(0)

    start_time = None
    alert_start_time = None
    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if detect_drowsiness(frame, model):
            if start_time is None:
                start_time = time.time()
            else:
                elapsed_time = time.time() - start_time
                if elapsed_time > 3 and not alert_triggered:
                    alert_start_time = time.time()
                    pygame.mixer.music.load('alert_sound1.mp3')  # Load alert sound
                    pygame.mixer.music.play(-1)  # Play alert sound in a loop
                    alert_triggered = True
                elif elapsed_time > 2:
                    start_time = None
                    alert_triggered = False
        else:
            start_time = None
            alert_triggered = False

            if alert_start_time is not None:
                elapsed_time = time.time() - alert_start_time
                if elapsed_time > 2:
                    pygame.mixer.music.stop()  # Stop playing alert sound
                    alert_start_time = None

        cv2.putText(frame, f"State: {'Drowsy' if alert_triggered else 'Alert'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
