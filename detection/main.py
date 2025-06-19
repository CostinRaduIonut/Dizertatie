from flask import Flask, request
from flask_restx import Api, Resource, reqparse
import numpy as np
import cv2 as cv
from werkzeug.datastructures import FileStorage 
import base64
from detection_cnn import process_image, model
from flask_cors import CORS
from utils.speech import generate_speech
from flask import send_file


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
upload_parser.add_argument(
    'outputType',
    location='form',
    type=str,
    choices=('image', 'sound'),
    required=True,
    help='Specify "image" or "sound" output'
)
@ns.route('/')
@api.expect(upload_parser)
class BrailleEndpoint(Resource):
    def post(self):
        args = upload_parser.parse_args()
        file = args['file']
        output_type = args['outputType']
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
        if output_type == 'sound':
            fname_voice = generate_speech(text)
            return send_file(fname_voice, as_attachment=True, download_name=fname_voice)
        elif output_type == 'image':
            return response

if __name__ == '__main__':
    app.run(debug=True)
