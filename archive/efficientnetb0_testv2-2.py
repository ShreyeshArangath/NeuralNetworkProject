import tensorflow as tf
from tensorflow import keras 
from keras import layers
from keras.applications.efficientnet import EfficientNetB0

import os 
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle

# Create pkl file of the model after the training phase
def dumpModel(modelName, phase): 
    # Save the trained model as a pickle string.
    modelName = "model_" + modelName + "_ " + phase + ".pkl"
    pickle.dump(model, open(modelName, 'wb'))
 
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
images = images.astype('float32')/255.0 # Normalize 
images.shape
#(6000, 224, 224, 3)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
y = df['Labels'].values
yLabelEncoder = LabelEncoder()
y = yLabelEncoder.fit_transform(y)

y = y.reshape(-1,1)

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y) #.toarray()
print(Y[:5])
print(Y[35:])

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

images, Y = shuffle(images,Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=127)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


NUM_CLASSES = 7
IMG_SIZE = 224
DROPOUT_RATE=0.2
#x = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
input = layers.Input(shape=(224,224,3))
baseModel = EfficientNetB0(
  include_top = False, 
  weights = 'imagenet', 
  classes = NUM_CLASSES)
baseModel.trainable = False
x = baseModel(input, training=False)
x =  layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.Dropout(DROPOUT_RATE, noise_shape=None, seed=None)(x)
output = layers.Dense(NUM_CLASSES,activation="softmax", name = "pred")(x)
model = keras.Model(input, output, name="FeatureExtraction-B0")
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy',        
        precision_score,
        recall_score,
        f1_score
    ]
)

#Feature extraction without top layers
hist_results = model.fit(
  train_x,
  validation_data=train_y,
  epochs=5
)

modelName="efficientNetB0"

dumpModel(modelName, "phase1")

# Fine tuning
baseModel.trainable = True
for layer in model.layers[1].layers:
    if isinstance(layer, layers.BatchNormalization):
      layer.trainable = False
        
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy',        
        precision_score,
        recall_score,
        f1_score
    ]
)

# Train it again 
hist_results_tuned = model.fit(
  train_x,
  validation_data=train_y,
  epochs=9,
  #steps_per_epoch=len(train_ds)?
  initial_epoch=hist_results.epoch[-1]
)


dumpModel(modelName, "phase2")

preds = model.predict(train_y, verbose = 1)
model.evaluate(train_y)

# model.summary()
# # # Train the model 
# hist = model.fit(train_x, train_y, epochs=10, verbose=2)

# # from sklearn.metrics import classification_report

# # y_pred = model.predict(test_x, batch_size=32, verbose=1)
# # y_pred_bool = np.argmax(y_pred, axis=1)

# # print(classification_report(y_test, y_pred_bool))

# def plot_hist(hist):
#     plt.plot(hist.history['accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(["train", "validation"], loc="upper left")
#     plt.show()

# plot_hist(hist)
# # # Evaluate the model 

# preds = model.evaluate(test_x, test_y)
# print ("Loss = " + str(preds[0]))
# print ("Test Accuracy = " + str(preds[1]))