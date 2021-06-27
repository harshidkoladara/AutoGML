from imutils.video import VideoStream
import imutils
import cv2
import os
import urllib.request
import numpy as np
from django.conf import settings
import face_recognition
from datetime import datetime
import pickle


class VideoCamera(object):
    def __init__(self, path):

        try:
            self.video = cv2.VideoCapture(0)
        except:
            self.video = cv2.VideoCapture(0)
        model = pickle.load(open(f'{path}/finalized_model.sav', 'rb'))
        self.encodeListKnown, self.className = model[0], model[1]

    def __del__(self):
        self.video.release()

    # This function is used in views

    def get_frame(self):

        success, image = self.video.read()

        imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25, )
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(
                self.encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(
                self.encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = self.className[matchIndex].upper()

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(image, (x1, y2 - 35), (x2, y2),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(image, name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


class IPWebCam(object):
    def __init__(self):
        self.url = "http://192.168.1.178:8080/shot.jpg"

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        imgResp = urllib.request.urlopen(self.url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        img = cv2.resize(img, (640, 480))
        frame_flip = cv2.flip(img, 1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()
