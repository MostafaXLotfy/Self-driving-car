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

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
class ResidualBlock(Layer):
  def __init__(self,conv1, conv2, conv3, conv4, bn1, bn2, bn3,**kwargs):
    super(ResidualBlock, self).__init__(**kwargs)
    self.conv1 = tf.keras.utils.deserialize_keras_object(
        identifier=conv1, module_objects=globals(),
         custom_objects=None,
        printable_module_name='man'
      )
    self.conv2 = tf.keras.utils.deserialize_keras_object(
        identifier=conv2, module_objects=globals(),
        custom_objects=None,
        printable_module_name='man'
      )
    self.conv3 = tf.keras.utils.deserialize_keras_object(
        identifier=conv3, module_objects=globals(),
        custom_objects=None,
        printable_module_name='man'
        )
    self.conv4 = tf.keras.utils.deserialize_keras_object(
        identifier=conv4, module_objects=globals(),
         custom_objects=None,
        printable_module_name='man'
      )  
    self.bn1 = tf.keras.utils.deserialize_keras_object(
        identifier=bn1, module_objects=globals(),
         custom_objects=None,
        printable_module_name='man'
      )
    self.bn2 = tf.keras.utils.deserialize_keras_object(
        identifier=bn2, module_objects=globals(),
         custom_objects=None,
        printable_module_name='man'
      )
    self.bn3 = tf.keras.utils.deserialize_keras_object(
        identifier=bn3, module_objects=globals(),
         custom_objects=None,
        printable_module_name='man'
      )
  def call(self, inputs):
    Y = self.conv1(relu(self.bn1(inputs)))
    Y = self.conv2(relu(self.bn2(Y)))
    Y = self.conv3(relu(self.bn3(Y)))
    if self.conv4 is not None:
      inputs = self.conv4(relu(inputs))
    return Y + inputs

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'conv1': self.conv1,
        'conv2': self.conv2,
        'conv3': self.conv3,
        'conv4': self.conv4,
        'bn1': self.bn1,
        'bn2': self.bn2,
        'bn3': self.bn3
    })
    return config

class ResidualStage(Layer):
  def __init__(self, blocks,**kwargs):
    super(ResidualStage, self).__init__(**kwargs)
    self.blocks = []
    for i in range(len(blocks)):
      self.blocks.append(tf.keras.utils.deserialize_keras_object(
        identifier=blocks[i], module_objects=None,
         custom_objects={'ResidualBlock':ResidualBlock},
          printable_module_name='man'
      ))
  
  def call(self, inputs):
    X = inputs
    for block in self.blocks:
      X = block(X)
    return X

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'blocks': self.blocks,
    })
    return config

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
        image_array = np.asarray(image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
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

    model = load_model('./v1.h5', custom_objects={
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
