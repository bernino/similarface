import cv2 
import face_recognition
import numpy as np
import pickle
import os
import glob
from imutils import build_montages
from imageai.Detection import ObjectDetection
import keras
from keras.models import load_model

known_face_encodings = []
known_face_metadata = []
imgs = []

model = load_model("model_v6_23.hdf5")
my_path = os.getcwd()
#detector = ObjectDetection()
#detector.setModelTypeAsRetinaNet()
#detector.setModelPath( os.path.join(my_path , "resnet50_coco_best_v2.0.1.h5"))
#detector.loadModel()

emotion_dict= {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
emotion_label_map = dict((v,k) for k,v in emotion_dict.items()) 

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,100)
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 1

# not being used for the moment
def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def get_imgs():
    for file in glob.glob('face/*.jpg'):
        imgs.append(file)
    return imgs


# build the face encodings 128d vectors and store in dict
imgs = get_imgs()
for i, img in enumerate(imgs):
    print("running "+ str(img))
    objectsfile = str(i)+"-objects.jpg"
    current_image = face_recognition.load_image_file(img)
    face_locations = face_recognition.face_locations(current_image)
    face_encodings = face_recognition.face_encodings(
                     current_image, 
                     face_locations)
    #detections = detector.detectObjectsFromImage(input_image=img, output_image_path=os.path.join(my_path , objectsfile))
    #for eachobj in detections:
    #    print(str(eachobj["name"]) + " : " + str(eachobj["percentage_probability"]) )

    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Grab the image of the the face
        top, right, bottom, left = face_location
        face_image = current_image[top:bottom, left:right]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = cv2.resize(face_image, (150, 150))
        face_imagetest = cv2.resize(face_image, (48,48))
        face_imagetest = cv2.cvtColor(face_imagetest, cv2.COLOR_BGR2GRAY)
        face_imagetest = np.reshape(face_imagetest, [1, face_imagetest.shape[0], face_imagetest.shape[1], 1])
        predicted_emotions = model.predict(face_imagetest)
        #predicted_emotions = np.argmax(model.predict(face_image))
        emotions = ""
        for (i,j), v in np.ndenumerate(predicted_emotions):
            emotion = emotion_label_map[j]
            emotions += str(emotion) + " - p:"+ str(v) + "\n"
        print(emotions)
        cv2.imshow("img",face_image)
        cv2.imwrite("out.jpg", face_image)
        cv2.waitKey(0)

        known_face_encodings.append(face_encoding)
        known_face_metadata.append({
            "encoding": face_encoding,
            "image": img,
            "face": face_image,
            "emotions": predicted_emotions
        })

# find the clusters from the face encodings
# output a montage of faces
for j, face in enumerate(known_face_encodings):
    faces = []
    print(j)
    face_distances = face_recognition.face_distance(known_face_encodings, face)
    original_face = known_face_metadata[j]["image"]
    original_face_pic = known_face_metadata[j]["face"]
    for i, face_distance in enumerate(face_distances):
        matching_face = known_face_metadata[i]["image"]
        if face_distance < 0.62:
            print("Added {} to {} with distance {}".format(face_distance, matching_face, original_face))
            image = known_face_metadata[i]["face"]
            faces.append(image)
    montage = build_montages(faces, (96, 96), (5, 5))[0]
    montagefile = str(j)+".jpg"
    cv2.imshow(montagefile, montage)
    cv2.imwrite(montagefile, montage)
    cv2.waitKey(0)