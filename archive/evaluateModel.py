import tensorflow as tf
from tensorflow import keras, sparse 
from keras import layers, applications
# from keras.preprocessing import image
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB5, EfficientNetB7
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.densenet import DenseNet121
# from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


import os 


# CHANGE: 
# imageSize: depends on pretrained model
# modelName
# trainingFolder: x5_RGB -> x5
# testingFolder: x5_RGB -> x5
# keras.Model(name='FeatureExtraction-<model name>')

# Data preprocessing 
from keras.preprocessing.image import load_img, img_to_array


preds = model.predict(val_ds, verbose = 1)
model.evaluate(val_ds)
