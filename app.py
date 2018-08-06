"""
1. Can't See Anything - Cloudy AF
2. Mostly cloudy or Foggy - I can almost see the Townsite
3. Lots of clouds or Fog - I can see the Townsite but some of the mountains aren't visible
4. Clouds are high in sky -  I can see the Townsite and most/all of the mountains
5. Barely any clouds or totally blue clear sky - Awesome View
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import urllib
import datetime
import random 

from utils import *
from threading import Thread
import tensorflow as tf


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

global GRAPH
target = "https://www.banffjaspercollection.com/BrewsterCMS/Handlers/Webcam-Photos.ashx?photo=wctown2"
model_file = r"C:\Users\vohprecio\Desktop\RateMyView\models\retrained_graph_5000.pb"
label_file = r"C:\Users\vohprecio\Desktop\RateMyView\models\retrained_labels.txt"
input_height = 299
input_width = 299
input_mean = 0
input_std = 255
input_layer = "Mul"
output_layer = "final_result"


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/your_model.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

def load_model():
    print('Model loaded')



@app.before_first_request
def initialize():
    thread = Thread(target=load_model)
    thread.daemon = True
    thread.start()



@app.route('/', methods=['GET'])
def index():
    # Main pages

    import logging

    logging.basicConfig(level=logging.DEBUG)


    def load_graph(model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        logging.debug("############ Graph ##################")
        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                    input_mean=0, input_std=255):
        input_name = "file_reader"
        output_name = "normalized"
        file_reader = tf.read_file(file_name, input_name)
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(file_reader, channels = 3,
                                           name='png_reader')
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
        else:
            image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                            name='jpeg_reader')
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0);
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def load_labels(label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    resource = urllib.request.urlopen(target)
    filename = gen_filename()
    file_name = r'C:\Users\vohprecio\Desktop\RateMyView\uploads' + '\\' + filename
    output = open(file_name,"wb")
    output.write(resource.read())
    output.close()
    GRAPH = load_graph(model_file)
    print()
    t = read_tensor_from_image_file(file_name,
                                      input_height=input_height,
                                      input_width=input_width,
                                      input_mean=input_mean,
                                      input_std=input_std)
   
    logging.debug("############ Reading File ##################")
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = GRAPH.get_operation_by_name(input_name)
    output_operation = GRAPH.get_operation_by_name(output_name)

    with tf.Session(graph=GRAPH) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
        end=time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    logging.debug(top_k[0])
    score = str(top_k[0] + 1)
    response = score + get_fortune(compliments, moar)
    return render_template('view.html', score=response)


@app.route('/hello/<name>', methods=['GET', 'POST'])
def hello(name):
     return 'Hello {}'.format(name)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
