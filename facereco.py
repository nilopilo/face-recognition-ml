import face_recognition
import pickle
import cv2
import base64
import os
import io
from flask import Flask,request,jsonify
import numpy as np
from PIL import Image

app=Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
    file = request.form.get('image')
    image = base64.b64decode(str(file))
    image = Image.open(io.BytesIO(image))
    image= np.array(image)
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    data = pickle.loads(open('face_enc (1)', "rb").read())
    data1 = pickle.loads(open('user_data (1)', "rb").read())

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    encodings = face_recognition.face_encodings(rgb)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            names.append(name)
            """for ((x, y, w, h), name) in zip(faces, names):
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)"""

    fin_name=max(set(names),key=names.count)
    index=data1["name"].index(fin_name)
    place=data1["place"][index]
    branch=data1["branch"][index]
    year=data1['year'][index]
    studying=data1['studying'][index]
    img = data1["image"][index]
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    result={'name':fin_name,'place':place,'branch':branch,'image':str(img_base64),'year':year}

    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True)

