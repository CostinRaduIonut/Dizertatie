from flask import Flask, request
from flask_restx import Api, Resource, reqparse
from flask_cors import CORS
import numpy as np
import cv2 as cv
from werkzeug.datastructures import FileStorage 
import base64
from detection import process_image, model 

app = Flask(__name__)
CORS(app, supports_credentials=True)
api = Api(app, doc="/docs")

ns = api.namespace('braille', description='Braille operations')

upload_parser = reqparse.RequestParser()
upload_parser.add_argument(
    'file',
    location='files',
    type=FileStorage, 
    required=True,
    help='Upload a Braille image file'
)

@ns.route('/')
@api.expect(upload_parser)
class BrailleEndpoint(Resource):
    def post(self):
        args = upload_parser.parse_args()
        file = args['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        text, img_result = process_image(img, model)
        _, buffer = cv.imencode('.png', img_result)
        result_base64 = base64.b64encode(buffer).decode("utf-8")
        response = {
            "text" : text,
            "img_base64" : result_base64
        }
        if img is None:
            return {"message": "Invalid image format."}, 400

        return response

if __name__ == '__main__':
    app.run(debug=True)
