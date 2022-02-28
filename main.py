from tensorflow import keras 
from keras import layers, applications
# from keras.preprocessing import image
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB5, EfficientNetB7
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.densenet import DenseNet121
# from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input

import os 
import numpy as np
import cv2
from keras.models import Sequential

# Data preprocessing 
from keras.preprocessing.image import load_img, img_to_array

# Load all the images from the folder 
dataFolder = os.path.dirname('data/x5_RGB/train/RGB/')
imagePaths = []
pil = []

isIgnoredFile = lambda x: x[0] == "."

for case in os.listdir(dataFolder):
    # print(case)
    if isIgnoredFile(case):
        continue

    f = os.path.join(dataFolder, case)
    for image in os.listdir(f):
        if isIgnoredFile(image):
            continue
        
        imagePath = os.path.join(dataFolder, case, image)
        imagePaths.append(imagePath)

# print(imagePaths[:10])

for imagePath in imagePaths:
    #img = load_img(imagePath) 
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (224,224))   
    x = img_to_array(img)    
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # print(x)
    pil.append(x)

# print(pil)

model = Sequential()

# Set up the layers 

efficientNetB0 = EfficientNetB0(weights="imagenet", include_top = False)   
# efficientNetB5 = EfficientNetB5(weights="imagenet", include_top=False)  
# efficientNetB7 = EfficientNetB7(weights="imagenet", include_top=False)
# resNet152V2 = ResNet152V2(weights='imagenet', include_top=False)
# denseNet121 = DenseNet121(weights="imagenet")

RATE = 0.2
UNITS = 0 

globalAveragePooling = layers.GlobalAveragePooling2D(data_format=None, keepdims=False)
dropout = layers.Dropout(RATE, noise_shape=None, seed=None)
dense = layers.Dense(
    UNITS,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)
softmax = layers.Softmax()

# Pretrained Models
model.add(efficientNetB0)
# model.add(efficientNetB5)
# model.add(efficientNetB7)
# model.add(resNet152V2)
# model.add(denseNet121)

# Additional layers
model.add(globalAveragePooling)
model.add(dropout)
model.add(dense)
model.add(softmax)

model.summary()
# Train the model 

# Evaluate the model 