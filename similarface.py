import cv2 
import face_recognition
import numpy as np
import pickle
import os
import glob
from imutils import build_montages
from imageai.Detection import ObjectDetection

known_face_encodings = []
known_face_metadata = []
imgs = []

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

imgs = get_imgs()
my_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(my_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

# build the face encodings 128d vectors and store in dict
for i, img in enumerate(imgs):
    print("running "+ str(img))
    objectsfile = str(i)+"-objects.jpg"
    current_image = face_recognition.load_image_file(img)
    face_locations = face_recognition.face_locations(current_image)
    face_encodings = face_recognition.face_encodings(
                     current_image, 
                     face_locations)
    detections = detector.detectObjectsFromImage(input_image=img, output_image_path=os.path.join(my_path , objectsfile))
    for eachobj in detections:
        print(str(eachobj["name"]) + " : " + str(eachobj["percentage_probability"]) )

    for face_location, face_encoding in zip(face_locations, face_encodings):
                       # Grab the image of the the face
                       top, right, bottom, left = face_location
                       face_image = current_image[top:bottom, left:right]
                       face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                       face_image = cv2.resize(face_image, (150, 150))

                       known_face_encodings.append(face_encoding)
                       known_face_metadata.append({
                           "encoding": face_encoding,
                           "image": img,
                           "face": face_image
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
            print("Added distance: {} of {} to  {}".format(face_distance, matching_face, original_face))
            image = known_face_metadata[i]["face"]
            faces.append(image)
    montage = build_montages(faces, (96, 96), (5, 5))[0]
    montagefile = str(j)+".jpg"
    cv2.imshow(montagefile, montage)
    cv2.imwrite(montagefile, montage)
    cv2.waitKey(0)