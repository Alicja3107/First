import cv2
from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
import requests

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)


class PeopleCounterStatic(Resource):
    def get(self):
        # load image
        image = cv2.imread('ludzie_przystanek.jpg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


class PeopleCounterDynamicUrl(Resource):
    def get(self):
        url = request.args.get('url')
        print('url', url)
        response = requests.get(url)
        image_data = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (700, 400))
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


class PeopleCounterPost(Resource):
    def post(self):
        file = request.files['file']
        image_data = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (700, 400))
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


api.add_resource(PeopleCounterStatic, '/')
api.add_resource(PeopleCounterDynamicUrl, '/dynamic')
api.add_resource(PeopleCounterPost, '/count_people')
if __name__ == '__main__':
    app.run(debug=True)
