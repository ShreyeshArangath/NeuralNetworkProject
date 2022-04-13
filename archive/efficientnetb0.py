import tensorflow as tf
from tensorflow import keras 
from keras import layers, applications
from keras.applications.efficientnet import EfficientNetB0

import os 
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
import pickle

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dumpModel(modelName, phase): 
    # Save the trained model as a pickle string.
    modelName = "model_" + modelName + "_ " + phase + ".pkl"
    pickle.dump(model, open(modelName, 'wb'))

# Data preprocessing 
 
# Load all the images from the folder 
dataFolder = os.path.dirname('data/x5/train/RGB/')
imagePaths = []
pil = []
classLabels = []
isIgnoredFile = lambda x: x[0] == "."

for case in os.listdir(dataFolder):
    # print(case)
    if isIgnoredFile(case):
        continue

    f = os.path.join(dataFolder, case)
    for image in os.listdir(f):
        if isIgnoredFile(image):
            continue
        label = f[-1]
        classLabels.append(label)
        imagePath = os.path.join(dataFolder, case, image)
        imagePaths.append(imagePath)
df = pd.DataFrame(data=zip(classLabels, imagePaths), columns=["Labels", "Images"])


IMAGE_SIZE = 224 
images = []
for imagePath in imagePaths:
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))   
    images.append(img)

images = np.array(images)
images = images.astype('float32')/255.0 #Normalize 
images.shape
#(6000, 224, 224, 3)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y = df['Labels'].values
yLabelEncoder = LabelEncoder()
y = yLabelEncoder.fit_transform(y)

from sklearn.compose import ColumnTransformer
y=y.reshape(-1,1)

ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y) #.toarray()
print(Y[:5])
print(Y[35:])

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
images, Y = shuffle(images,Y, random_state=1)

train_x, test_x, train_y,test_y = train_test_split(images, Y, test_size=0.05, random_state=127)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


NUM_CLASSES = 7
IMG_SIZE = 224
DROPOUT_RATE = 0.2
UNITS = 0 

modelName = "efficientNetB0"
inpShape =  (IMG_SIZE, IMG_SIZE, 3)
inp = layers.Input(shape=inpShape)
baseModel = EfficientNetB0(
    weights="imagenet", 
    include_top = False
)   

baseModel.trainable = False 
x = baseModel(inp, training=False)
x =  layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.Dropout(
    DROPOUT_RATE, 
    noise_shape=None, 
    seed=None
)(x)
out = layers.Dense(
    NUM_CLASSES,
    activation="softmax", 
    name = "pred"
)(x)
model = keras.Model(inp, out, name="FeatureExtraction-B0")
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        'accuracy',
        recall_m,
        precision_m,
        f1_m
    ]
)

model.summary()

# Train the model 
hist_results = model.fit(
    train_x, 
    train_y, 
    epochs=5, 
    verbose=2
)

try:
    dumpModel(modelName, "phase1")
except Exception:
    pass


baseModel.trainable = True
for layer in model.layers[1].layers:
    if isinstance(layer, layers.BatchNormalization):
      layer.trainable = False
        
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[
        'accuracy',
        recall_m,
        precision_m,
        f1_m
    ]
)

# Train it again 
hist_results_tuned = model.fit(
  train_x,
  train_y,
  epochs=9,
  steps_per_epoch=len(train_y),
  initial_epoch=model.epoch[-1]
)

try:
    dumpModel(modelName, "phase2")
except Exception:
    pass

###
preds = model.evaluate(test_x, test_y)
# preds = model.predict(test_y, verbose = 1)

from sklearn.metrics import classification_report

y_pred = model.predict(test_x, batch_size=32, verbose=1)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(test_y, y_pred_bool))
