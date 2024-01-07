import cv2
import face_recognition
import os

data_folder = 'test_images'

known_faces = {}
known_names = []

for person_folder in os.listdir(data_folder):
    person_path = os.path.join(data_folder, person_folder)

    if os.path.isdir(person_path):
        face_encodings = []

        for filename in os.listdir(person_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(person_path, filename)
                img = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(img)

                if encoding:
                    face_encodings.append(encoding[0])

        if face_encodings:
            known_faces[person_folder] = face_encodings
            known_names.append(person_folder)

# Initialize the face cascade
face_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_body = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haarcascade classifier
    faces = face_cascade_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect full bodies using the Fullbody classifier
    bodies = face_cascade_body.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_image = frame[y:y + h, x:x + w]

        # Encode the face using face_recognition
        face_encodings = face_recognition.face_encodings(face_image)

        # Check if any face encodings are found
        if face_encodings:
            # Compare the face encoding with known faces
            match_found = False
            for name in known_names:
                for known_face_encoding in known_faces[name]:
                    results = face_recognition.compare_faces([known_face_encoding], face_encodings[0])
                    if True in results:
                        # Draw a rectangle around the face and display the name
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        match_found = True
                        break
                if match_found:
                    break

            # If no match is found, display "Unknown"
            if not match_found:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Iterate over detected full bodies
    for (x, y, w, h) in bodies:
        # Draw a rectangle around the full body
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
