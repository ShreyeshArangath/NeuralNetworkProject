# CHANGE: 
# from keras.applications.<model> import <modelname>
# imageSize: (x, x); x = 600 if efficientnetb7, 456 if efficientnetb5, 224 for others
# Size & batch_size: 64 if ResNet, 32 for others
# modelName: <modelname>
# inpShape: (x, x, 3); x = 600 if efficientnetb7, 456 if efficientnetb5, 224 for others
# trainingFolder: x5_RGB -> x5
# testingFolder: x5_RGB -> x5
# keras.Model(name='FeatureExtraction-<model name>')  
# baseModel = <modelname>( <inputs> )

# import os

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications.efficientnet import EfficientNetB7
from keras import backend as K
# from keras.backend import sigmoid
# from keras.utils.generic_utils import get_custom_objects
# from keras.layers import Activation

import pickle

# Retrieve data from dataFolder
def getDataset(dataFolder, imageSize = (600, 600), batchSize = 32):
    train_ds = keras.utils.image_dataset_from_directory(
      dataFolder,
      seed=0,
      image_size=imageSize,
      batch_size=batchSize
    )
    return train_ds

# Tune buffer size and efficiency 
def configurePerformance(train_ds, val_ds): 
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

# # Define swish function
# def swish(x, beta = 1):
#     return (x * sigmoid(beta * x))

# Calculate recall
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    # print(f"true positives: {true_positives}       possible positives:  {possible_positives}         recall:    {recall}")
    return recall

# Calculate precision
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    # print(f"true positives: {true_positives}       predicted positives: {predicted_positives}        precision: {precision}")
    return precision
    
# Calculate f1 score
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)    
    f1_score = 2*((precision*recall)/(precision+recall+K.epsilon()))
    # print(f"f1score:        {f1_score}")
    return f1_score

# Create pkl file of the model after the training phase
def dumpModel(modelName, phase): 
    # Save the trained model as a pickle string.
    modelName = "model_" + modelName + "_ " + phase + ".pkl"
    pickle.dump(model, open(modelName, 'wb'))

def getDatasetsByCar(cars):
    train_ds = None 
    val_ds = None 
    for car in cars: 
        trainingFolder = 'data/'+ car +'/train/RGB/'
        testingFolder = 'data/'+ car + '/test_with_labels/RGB/'
        if not train_ds and not val_ds:
            train_ds = getDataset(trainingFolder)
            val_ds =  getDataset(testingFolder)
        else: 
            new_train_ds =  getDataset(trainingFolder)
            new_val_ds =  getDataset(testingFolder)
            train_ds.concatenate(new_train_ds)
            val_ds.concatenate(new_val_ds)
    return train_ds, val_ds

EPOCHS = 5
modelName = "efficientNetB7"

# Initial layer input shape
inpShape =  (600, 600, 3)
cars = ['x5', 'model3', 'hilux']

# Retrieve training and testing data
train_ds, val_ds = getDatasetsByCar(cars)
train_ds, val_ds = configurePerformance(train_ds, val_ds)

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(inpShape[0], inpShape[0]),
  layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

dropoutRate = 0.2
numClasses = 7 
inp = layers.Input(shape=inpShape)
baseModel = EfficientNetB7(
    weights = "imagenet",
    include_top = False,
    classes = numClasses
) 
augmentation = tf.keras.Sequential([
  # Add the preprocessing layers you created earlier.
  resize_and_rescale,
  data_augmentation,
])

inp = augmentation(inp)

# # Creating a Custom object to use the swish activation fn
# get_custom_objects().update({'swish': Activation(swish)})

baseModel.trainable = False 
x = baseModel(inp, training=False)
x =  layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.Dropout(dropoutRate, noise_shape=None, seed=None)(x)
out = layers.Dense(numClasses, activation="softmax", name = "pred")(x)
model = keras.Model(inp, out, name="FeatureExtraction-EfficientNetB7")

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy',        
        precision_m,
        recall_m,
        f1_m
    ]
)

# Train using feature extraction without top layers 
hist_results = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS,
#   steps_per_epoch=len(train_ds),
#   batch_size=64
)

try:
    dumpModel(modelName, "phase1")
except Exception:
    pass

# Fine tune the Feature Extraction Model 
baseModel.trainable = True
for layer in model.layers[1].layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
        
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy',        
        precision_m,
        recall_m,
        f1_m
    ]
)

# Train again using the fine-tuned model
hist_results_tuned = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=9,  
  initial_epoch=hist_results.epoch[-1],
#   steps_per_epoch=len(train_ds),
#   batch_size=64
)

try:
    dumpModel(modelName, "phase2")
except Exception:
    pass

preds = model.predict(val_ds, verbose = 1)
model.evaluate(val_ds)