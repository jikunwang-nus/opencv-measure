#!flask/bin/python
import os
import coreCV
from flask import Flask, jsonify, json, send_from_directory
from flask import request

import image
from result import Rt

# import json
app = Flask(__name__)
# app.config.from_file("config.json", load=json.load)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]


@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})


@app.route('/image/save', methods=['POST'])
def save_image():
    file = request.files['file']
    file_name = file.filename
    file_pre_path = os.getcwd() + '/temp/' + file_name
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
    app.run(debug=True, port=5000)
