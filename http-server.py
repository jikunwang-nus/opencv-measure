#!flask/bin/python
import os

from flask import Flask, jsonify, send_from_directory
from flask import request

import coreCV
import image
from result import Rt

# import json
app = Flask(__name__)


# app.config.from_file("config.json", load=json.load)


@app.route('/ping', methods=['GET'])
def ping():
    return 'Pong'


@app.route('/image/save', methods=['POST'])
def save_image():
    file = request.files['file']
    file_name = file.filename
    file_pre_path = os.getcwd() + '/' + file_name
    if file:
        file.save(file_pre_path)
        coreCV.handle_entry(file_pre_path)
        res = Rt.success
        images = ["original-after-bd.png", 'original-after-bd-red.png']
        base64_res = []
        for i in images:
            base64_res.append(image_view(i))
        res['images'] = base64_res
        return jsonify(res)
    else:
        return jsonify(Rt.fail)


@app.route('/image/view', methods=['GET'])
def image_view(file_name):
    image_path = os.getcwd() + '/' + file_name
    stream = image.return_img_stream(image_path)
    return 'data:image/png;base64,' + stream


@app.route('/files/download', methods=["GET"])
def download_file():
    directory = os.getcwd()
    return send_from_directory(directory, '偏离值汇总.xlsx', as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
