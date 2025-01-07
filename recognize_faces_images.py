import face_recognition
import pickle
import cv2
import argparse

ap=argparse.ArgumentParser()

ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-d", "--detection-method", type= str, default="cnn",
                help="face detection model to use: either hog or cnn")
args=vars(ap.parse_args())  #detection-method is automatically converted to detection_method when using vars(ap.parse_args)


#load the pickle file 
print("[INFO] Loading encodings")
data= pickle.loads(open(args["encodings"], "rb").read())

#laod the input image and then convert it from BGR to RGB

image=cv2.imread(args["image"])
rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#detect the (x, y) co-ordinates of the bounding boxes corresponding to each face in the input image
#Then compute the facial embeddings for each face

print("[INFO] Recognizing Faces..")
boxes=face_recognition.face_locations(rgb, model=args["detection_method"]) 

encodings=face_recognition.face_encodings(rgb, boxes)

#initialize the list of names for each face detected
names=[]

for encoding in encodings:
    matches=face_recognition.compare_faces(data["encodings"], encoding)
    name="Unknown" #if faces dont match in database, it will be labeled as unknown

#the encodings of the input image we provided attempts to match the encoding of our known encoding dataset
#the output of this is a bunch of True/False

#The compare_faces computes the Euclidian distance between the candidate embedding and all faces in our dataset
#if the distance is below some threshold value, then its true. If not then false.
#it is more of a fancy way of implementing the k-NN model.

if True in matches:
    matchedIdx=[i for (i, b) in enumerate(matches) if b]
    '''for (i, b) in enumerate(matches):
            if b is True:
                matchesIdx.append(i)'''
    
    counts={} #dictionary

    for i in matchedIdx:
        name=data["names"][i]
        counts[name]=counts.get(name, 0)+1

    name=max(counts, key=counts.get)

names.append(name)
    

#to loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0,255,0),2)
    y=top-15 if top-15>15 else top+15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0),2)

cv2.imshow("Image", image)
cv2.waitKey(0)


