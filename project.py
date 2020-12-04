#=======================

# During the live testing, Haar Cascades [15] were used to identify a face. 
# This identified face was then taken as an image, converted to gray-scale and downscaled to a 48*48 image.
# Thus the image was converted to a format identical to that which was used to train the model.

import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
CASC_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.cv2.CascadeClassifier(CASC_PATH)


def face_dect(image):
  """
  Detecting faces in image
  :param image: 
  :return:  the coordinate of max face
  """
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  if not len(faces) > 0:
    return None
  max_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_face[2] * max_face[3]:
      max_face = face
  face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
  try:
    image = cv2.cv2.resize(face_image, (48, 48), interpolation=cv2.cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+} Problem during resize")
    return None
  return True




cap = cv2.cv2.VideoCapture(0)
i = 0
while(True):
 
    ret, frame = cap.read()
    gray = cv2.cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray, 1.3, 5)
    # cv2.cv2.imshow('frame',)
    # name = ' ' + str(i) + '.jpg'
    # print ('Creating...' + name)
    # cv2.cv2.imwrite(name, frame)
    for (x,y,w,h) in faces:
        print("Found Face")
        iface =  frame[int(y):int(y+h), int(x):int(x+w)]
        iface = cv2.cv2.cvtColor(iface, cv2.cv2.COLOR_BGR2GRAY)
        iface = cv2.cv2.resize(iface, (48, 48))
        
        # cv2.cv2.imshow('frame',frame)
        img_pixels = image.img_to_array(iface)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        model = tf.keras.models.load_model("cnn1.model")
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion = emotions[max_index]
        cv2.cv2.putText(frame, emotion, (int(x), int(y)), cv2.cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.cv2.imshow('frame',frame)

        
    else:
        print("no face")
    # resized_image = cv2.cv2.resize(gray,(48,48))
    # cv2.cv2.imshow('frame',resized_image)

    
    # model = tf.keras.models.load_model("cnn1.model")
    # prediction = model.predict(resized_image)
    # print(prediction)
    # # temp.append(np.argmax(prediction))
    # i+=1
    if cv2.cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.cv2.destroyAllWindows()



# import tensorflow as tf
# from tensorflow import keras


#==================== TO USE THE SAVED MODEL =======================
# model = tf.keras.models.load_model("neuralnet12.model")
#==================== TO MAKE PREDICTION ===========================
# prediction = model.predict([prepare(no)])
#         print(prediction)
#         temp.append(np.argmax(prediction))