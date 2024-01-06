import streamlit as st
import face_recognition
import os

st.title("Face Recognition with Streamlit")

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

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the uploaded image
    img = face_recognition.load_image_file(uploaded_file)

    # Find all face locations and face encodings in the uploaded image
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        match_found = False
        for name in known_names:
            for known_face_encoding in known_faces[name]:
                results = face_recognition.compare_faces([known_face_encoding], face_encoding)
                if True in results:
                    st.image(img[top:bottom, left:right], caption=f"Match: {name}", channels="RGB", use_column_width=True)
                    match_found = True
                    break
            if match_found:
                break

        # If no match is found, display "Unknown"
        if not match_found:
            st.image(img[top:bottom, left:right], caption="Unknown", channels="RGB", use_column_width=True)
