# For a in-depth description of the ipynb notebook please check the Code section in the report

# Importing needed libraries

from pickle import NONE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os

## Keras
import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

import cv2
import pandas as pd
import random
import ntpath
import shutil

## Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from imgaug import augmenters as aug
from tensorflow.keras.models import save_model
import warnings


# WAS INITIALLY USED FOR SPLITTING THE DATA NO LONGER NEEDED
#TRAIN_TRACK_2 = True #False for track 1 

## Directories
datadir = './Data'
imgdir = './Data/IMG'
data = NONE

# Get Filenames
def path_leaf(path):
  """Get tail of path"""
  head, tail = ntpath.split(path)
  return tail

# WAS INITIALLY USED FOR SPLITTING THE DATA NO LONGER NEEDED

# from datetime import datetime

#     #center_2022_05_15_21_00_09_631.jpg FIRST TRACK 2
# if (TRAIN_TRACK_2):
#     test_list = data['center'].tolist()
#     boolean_list = []
#     first_track2 = datetime(2022,5,15,21,0,9)
#     print(first_track2)

#     for index in range(len(test_list)):
#         test_list[index] = test_list[index][7:-8].replace('_',' ')
#         test_list[index] = datetime.strptime(test_list[index], '%Y %m %d %H %M %S')
#         if(test_list[index] >= first_track2):
#             boolean_list.append(True)
#         else:
#             boolean_list.append(False)
        
#     data = (data[boolean_list])
#     data.reset_index(drop=True, inplace=True)
# data.head()


# Load images into and angles into two numpy arrays
def load_img_steering():
  global imgdir, data, datadir
  """Get img and steering data into arrays"""
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(imgdir, center.strip()))
    steering.append(float(indexed_data[3]))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings


# Add image augmentation to the training data
def augment_image(path, steering_angle):
    img = npimg.imread(path)
    if np.random.rand() < 0.5:
        pan = aug.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = aug.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = aug.Multiply((0.4,1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering_angle *= -1
    
    return img, steering_angle
    

# Preprocess both training and validation images
def img_preprocess(img):
  """Take in path of img, returns preprocessed image"""
  #img = npimg.imread(img)
  
  ## Crop image to remove unnecessary features
  img = img[60:135, :, :]
  
  ## Change to YUV image
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  
  ## Gaussian blur
  img = cv2.GaussianBlur(img, (3, 3), 0)
  
  ## Decrease size for easier processing
  img = cv2.resize(img, (200, 66))
  
  ## Normalize values
  img = img / 255
  return img

# This function generates batches of training and validation images to be used in fitting the model 
def generate_batch(image_paths, steerings, batch_size, training_flag):
    while True:
        image_batch = []
        steerings_batch = []
        
        for i in range(batch_size):
            index = random.randint(0,len(image_paths)-1)
            if training_flag:
                img, steering = augment_image(image_paths[index],steerings[index])
            else:
                img = npimg.imread(image_paths[index])
                steering = steerings[index]
            img = img_preprocess(img)
            image_batch.append(img)
            steerings_batch.append(steering)
        yield(np.asarray(image_batch),np.asarray(steerings_batch))


# Builds the model proposed in the NVIDIA paper
def nvidia_model():
    model = Sequential()
    
    model.add(Convolution2D(23, (5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36, (5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48, (5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64, (3,3),(1,1),activation='elu'))
    model.add(Convolution2D(64, (3,3),(1,1),activation='elu'))
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(50,activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    
    return model


def main():
    global data, datadir, imgdir
    warnings.filterwarnings("ignore")

    # Loading the data
    print('start loading data')
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
    pd.set_option('display.max_colwidth', -1)

    # Getting the filenames of the images
    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)

    # Balancing the data to remove some data from bins which are above a certain threshold
    print('start balancing data')
    num_bins = 31
    samples_per_bin = 1000
    hist, bins = np.histogram(data['steering'], num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5  # center the bins to 0

    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(data['steering'])):
            steering_angle = data['steering'][i]
            if steering_angle >= bins[j] and steering_angle <= bins[j+1]:
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)
    
    
    data.drop(data.index[remove_list], inplace=True)

    # Load images and angles into two numpy arrays
    image_paths, steerings = load_img_steering()

    # Split the data into training and validation sets with a proportion of 80% to 20% respectively
    X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=42, shuffle=True)

    # Create the model
    print('start training')
    model = nvidia_model()   

    # Adding callbacks to create checkpoints for the model
    checkpoint_filepath = './checkpoint'
    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_freq = "epoch",
    save_best_only=True) 

    # Start training the model
    history = model.fit(generate_batch(X_train,Y_train,100,1), steps_per_epoch=300, epochs=30,
    validation_data=generate_batch(X_valid, Y_valid,10,0),validation_steps=200,  callbacks=[model_checkpoint_callback])
    
    # Load the best checkpoint and save the model as an .h5 file
    print('saving model')
    model.load_weights(checkpoint_filepath)
    model.save('./model.h5',save_format='h5')

if __name__ == "__main__":
    main()
