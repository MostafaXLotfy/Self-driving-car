import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras.layers import Layer, Conv2D, BatchNormalization
from keras import __version__ as keras_version

from keras.activations import relu
import tensorflow as tf
import cv2


def deserialize(dictionary, name, module_objects=globals(), custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        identifier=dictionary, module_objects=module_objects,
         custom_objects=custom_objects,
        printable_module_name=name
      )

def preprocess(image, size):
  print(np.shape(image))
  return (cv2.resize(image, size) / 255 - 0.5) / 0.5


class ResidualBlock(Layer):
  def __init__(self,conv1, conv2, conv3, conv4, bn1, bn2, bn3,**kwargs):
    super(ResidualBlock, self).__init__(**kwargs)
    self.conv1 = deserialize(conv1, f'{self.name}_conv1')
    self.conv2 = deserialize(conv2, f'{self.name}_conv2')
    self.conv3 = deserialize(conv3, f'{self.name}_conv3')
    self.conv4 = deserialize(conv4, f'{self.name}_conv4')
    self.bn1 = deserialize(bn1, f'{self.name}_bn1')
    self.bn2 = deserialize(bn2, f'{self.name}_bn2')
    self.bn3 = deserialize(bn3, f'{self.name}_bn3')

  def call(self, inputs):
    Y = self.conv1(relu(self.bn1(inputs)))
    Y = self.conv2(relu(self.bn2(Y)))
    Y = self.conv3(relu(self.bn3(Y)))
    if self.conv4 is not None:
      inputs = self.conv4(relu(inputs))
    return Y + inputs

class ResidualStage(Layer):
  def __init__(self, blocks,**kwargs):
    super(ResidualStage, self).__init__(**kwargs)
    self.blocks = []
    for i in range(len(blocks)):
      self.blocks.append(deserialize(blocks[i], f'{self.name}_block{i}',custom_objects={'ResidualBlock':ResidualBlock}))
  
  def call(self, inputs):
    X = inputs
    for block in self.blocks:
      X = block(X)
    return X


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = preprocess(np.asarray(image), size=(100, 100))
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
        # try: 
        #   image = np.asarray(image)
        #   print(image.shape)
        #   image = preprocess(image, (100, 100))
        # except Exception as e:
        #   print(e)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model, custom_objects={
        'ResidualStage':ResidualStage
        })

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
