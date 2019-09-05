import cv2 
import face_recognition
import numpy as np
import pickle
import os
import glob
from imutils import build_montages
#from imageai.Detection import ObjectDetection
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import sqlite3
import json
import cvlib as cv
from cvlib.object_detection import draw_bbox

# inspired from https://github.com/omar178/Emotion-recognition/blob/master/real_time_video.py

# initialisation and general setup 
known_face_encodings = []
known_face_metadata = []
imgs = []
imgs2 = []

emotion_model_path = '_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
emotions = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

#my_path = os.getcwd()
#detector = ObjectDetection()
#detector.setModelTypeAsRetinaNet()
#detector.setModelPath( os.path.join(my_path , "resnet50_coco_best_v2.0.1.h5"))
#detector.loadModel()

#font                   = cv2.FONT_HERSHEY_SIMPLEX
#bottomLeftCornerOfText = (10,100)
#fontScale              = 0.5
#fontColor              = (255,255,255)
#lineType               = 1

def sql_connection():
    try:
        conn = sqlite3.connect('similarfacesqlite.db')
        return conn
    except Error:
        print(Error)

conn = sql_connection()
#conn.set_trace_callback(print)
cur = conn.cursor()

def get_imgs():
    for file in glob.glob('faces/*.*'):
        imgs.append(file)
    return imgs


def match_imgs():
    for file2 in glob.glob('tomatch/*.*'):
        imgs2.append(file2)
    return imgs2


def get_encoding(test_img, cur):
    select = 'SELECT * from faces where imagepath = "'+str(test_img)+'"'
    cur.execute(select)
    data = cur.fetchone()
    if data:
        return data
    else:
        return False


def pickleload(subject):
    pickleload = pickle.loads(subject)
    return pickleload


def pickledump(subject):
    pickled = pickle.dumps(subject)
    return pickled


# build the face encodings 128d vectors and store in db
# detect emotions and store in db
# detect objects and store in db
imgs = get_imgs()
for i, img in enumerate(imgs):
    data = get_encoding(img, cur)
    #objectsfile = str(img)+"-objects.jpg"
    if not data:
        print("encoding "+ str(img))
        current_image = face_recognition.load_image_file(img)
        face_locations = face_recognition.face_locations(current_image)
        face_encodings = face_recognition.face_encodings(
                         current_image, 
                         face_locations)
        image = cv2.imread(img)
        bbox, label, conf = cv.detect_common_objects(image)
        #detections = detector.detectObjectsFromImage(input_image=img, output_image_path=os.path.join(my_path , objectsfile))
        detections = dict(label=label, conf=conf, bbox=bbox)
        #for m, things in enumerate(detections):
        #    # this overwrites detections and adds the array as a list
        #    listed = things['box_points'].tolist()
        #    things['box_points']=listed
        objects_detected_json = json.dumps(detections)

        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Grab the image of the the face
            # store everything in db
            top, right, bottom, left = face_location
            face_image = current_image[top:bottom, left:right]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = cv2.resize(face_image, (150, 150))
            ## setting up for emotion detection
            face_imagetest = cv2.resize(face_image, (64,64))
            face_imagetest = cv2.cvtColor(face_imagetest, cv2.COLOR_BGR2GRAY)
            face_imagetest = face_imagetest.astype("float") / 255.0
            face_imagetest = img_to_array(face_imagetest)
            face_imagetest = np.expand_dims(face_imagetest, axis=0)
            predicted_emotions = emotion_classifier.predict(face_imagetest)[0]
            #max_emotion_probability = np.max(predicted_emotions)
            #max_emotion = emotions[predicted_emotions.argmax()]
            predicted_emotions = predicted_emotions.tolist()
            #emotionstring = ""
            #for j, v in enumerate(predicted_emotions):
            #    emotion = emotions[j]
            #    emotionstring += str(emotion) + " - p:"+ str(v) + "\n"
            #emotionstring += "mostly " + str(max_emotion) + " with prob: " + str(max_emotion_probability)
            #print(emotionstring)
            #cv2.imshow(img,face_image)
            #cv2.waitKey(0)

            # pickling all the variables 
            face_image_pickled = pickledump(face_image)
            face_encoding_pickled = pickledump(face_encoding)
            face_location_pickled = pickledump(face_location)

            entities = (objects_detected_json, face_location_pickled, face_encoding_pickled, img, face_image_pickled, 
                    predicted_emotions[0], predicted_emotions[1] , predicted_emotions[2], predicted_emotions[3],
                    predicted_emotions[4], predicted_emotions[5], predicted_emotions[6])
            cur.execute("INSERT INTO faces(objects, face_location, face_encoding, imagepath, face, angry, disgust, fear, happy, sad, surprise, neutral) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", entities)
            conn.commit()
    else:
        print("using already done data for {}".format(data[3]))
        face_encoding = pickleload(data[2])
        face_location_pickled = pickleload(data[1])
        face_image = pickleload(data[4])
        # todo - get all emotions as well though not used

    known_face_encodings.append(face_encoding)
    known_face_metadata.append({
        "encoding": face_encoding,
        "image": img,
        "face": face_image,
    })
        #"emotions": predicted_emotions


# find the clusters from the face encodings
# output a montage of faces
# be polite to the db
conn.close()
imgs_to_match = match_imgs()
for k, img2 in enumerate(imgs_to_match):
    current_image2 = face_recognition.load_image_file(img2)
    face_locations2 = face_recognition.face_locations(current_image2)
    face_encodings2 = face_recognition.face_encodings(
                     current_image2, 
                     face_locations2)
    m = 0
    for face_location2, face_encoding2 in zip(face_locations2, face_encodings2):
        print("matching face {}".format(img2))
        faces = []
        #original_face = known_face_metadata[j]["image"]
        #original_face_pic = known_face_metadata[j]["face"]
        top, right, bottom, left = face_location2
        face_image2 = current_image2[top:bottom, left:right]
        face_image2 = cv2.cvtColor(face_image2, cv2.COLOR_BGR2RGB)
        face_image2 = cv2.resize(face_image2, (150, 150))
        faces.append(face_image2)
        for f, faces_encoded in enumerate(known_face_encodings):
            face_distances = face_recognition.face_distance([faces_encoded], face_encoding2)
            for i, face_distance in enumerate(face_distances): 
                matching_face = known_face_metadata[f]["image"] 
                if face_distance < 0.62:
                    print("Added {} to {} with distance {}".format(matching_face, img2, face_distance))
                    image = known_face_metadata[f]["face"]
                    faces.append(image)
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        montagefile = str(m)+".jpg"
        m += 1
        cv2.imshow(montagefile, montage)
        cv2.imwrite(montagefile, montage)
        cv2.waitKey(0)