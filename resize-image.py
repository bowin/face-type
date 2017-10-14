from PIL import Image, ImageDraw
import os
import face_recognition
import csv
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

def faceRecognition(id, label, img_path):
  image = ''
  try:
    image = face_recognition.load_image_file(img_path)
  except: 
    return
  face_locations = face_recognition.face_locations(image)
  top, right, bottom, left = face_locations[0]
  face_landmarks= face_recognition.face_landmarks(image)[0]
  #face_image = image[top:bottom, left:right]
  #pil_image = Image.fromarray(face_image)
  pil_image = Image.fromarray(image)
  d = ImageDraw.Draw(pil_image)
  for facial_feature in facial_features:
	  d.line(face_landmarks[facial_feature], width=5)
  pil_image.crop((left, top, right + 20, bottom + 20)).resize((128,128)).save('image-train/{label}-{id}.jpg'.format(label=label, id=id))


def genFaceFromImage(id, label):
  for x in range(10):
    img_path = 'face-db/{id}/{id}-00{x}.bmp'.format(id=id, x=x+1)
    faceRecognition('{id}{x}'.format(x=x, id=id), label, img_path)

with open('labeled.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
  faces = []
  for index, row in enumerate(spamreader):
    if (index == 0) : 
      faces = [x.strip() for x in row[1:]]
      print faces
      continue
    maxLen = 1
    maxIndex = -1
    for i, length in enumerate([len(x.strip()) for x in row[1:]]):
      if (length > maxLen):
        maxLen = length
        maxIndex = i
    if maxIndex == -1:
      continue   
    id = ''
    if index < 10:
      id = '00' + str(index)    
    elif index < 100:
      id = '0' + str(index)
    else:
      id = str(index)
    print id, maxIndex
    genFaceFromImage(id, maxIndex)
      
