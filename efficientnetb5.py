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
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential

# Data preprocessing 
from keras.preprocessing.image import load_img, img_to_array

def getDataset(dataFolder, subset, imageSize = (224, 224), batchSize = 32):
    train_ds = keras.utils.image_dataset_from_directory(
      dataFolder,
      seed=123,
      image_size=imageSize,
      batch_size=batchSize)
    return train_ds


def configurePerformance(train_ds, val_ds): 
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

# def buildModel(dropoutRate, numClasses, inpShape = (224, 224, 3)):
import pickle

def dumpModel(modelName, phase): 
    # Save the trained model as a pickle string.
    modelName = "model_" + modelName + "_ " + phase + ".pkl"
    pickle.dump(model, open(modelName, 'wb'))

EPOCHS = 5
modelName = "efficientNetB5"
inpShape =  (456, 456, 3)
trainingFolder = 'data/x5/train/RGB/'
testingFolder = 'data/x5/test_with_labels/RGB/'
train_ds = getDataset(trainingFolder, "training")
val_ds =  getDataset(testingFolder, "validation")

dropoutRate = 0.2
numClasses = 7 
inp = layers.Input(shape=inpShape)
baseModel = EfficientNetB0(weights="imagenet",
                   include_top = False) 

baseModel.trainable = False 
x = baseModel(inp, training=False)
x =  layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.Dropout(dropoutRate, noise_shape=None, seed=None)(x)
out = layers.Dense(numClasses,activation="softmax", name = "pred")(x)
model = keras.Model(inp, out, name="FeatureExtraction-B5")
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=['accuracy'])

# Feature extraction without the top layers 
hist_results = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

dumpModel(modelName, "phase1")


# Train it again 
hist_results_tuned = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

dumpModel(modelName, "phase2")
