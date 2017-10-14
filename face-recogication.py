from PIL import Image, ImageDraw
import os
import face_recognition

image = face_recognition.load_image_file('face-db/001/001-001.bmp')
face_locations = face_recognition.face_locations(image)
top, right, bottom, left = face_locations[0]
print face_locations[0]
face_landmarks= face_recognition.face_landmarks(image)[0]
print face_landmarks
facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]

face_image = image[top:bottom, left:right]
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)
for facial_feature in facial_features:
	d.line(face_landmarks[facial_feature], width=5)
pil_image.crop((left, top, right + 20, bottom + 20)).resize((128,128)).save('faceR-3.jpg')

