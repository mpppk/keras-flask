import os

from flask import Flask, redirect, request, jsonify
from keras import models
import numpy as np
import tensorflow as tf
from PIL import Image
import io


app = Flask(__name__)
# model = None
model = models.load_model('sutaba-model.h5')
model.summary()
graph = tf.get_default_graph()

classes = [
    'sutaba',
    'ramen',
    'other',
]


def load_model():
    global model
    model = models.load_model('sutaba-model.h5')
    model.summary()


@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'file' in request.files:
        img = request.files['file'].read()
        img = Image.open(io.BytesIO(img))
        img.save('test.jpg')
        img = np.asarray(img) / 255.
        img = np.expand_dims(img, axis=0)
        global graph
        with graph.as_default():
            pred = model.predict(img)
            confidence = str(round(max(pred[0]), 3))
            pred = classes[np.argmax(pred)]

            data = dict(pred=pred, confidence=confidence)
            return jsonify(data)
    return '{error: "image does not exist in file key"}'


if __name__ == '__main__':
    # load_model()
    app.run(debug=True, host="0.0.0.0", port=os.environ.get('PORT', 5000))

