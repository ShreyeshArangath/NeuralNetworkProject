import tensorflow as tf
from tensorflow import keras
from keras import layers
# from keras.preprocessing import image
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.vgg16 import VGG16
import tensorflow_addons as tfa

from keras import backend as K

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

# Retrieve training data
# remove subset param
def getDataset(dataFolder, subset, imageSize = (224, 224), batchSize = 32):
    train_ds = keras.utils.image_dataset_from_directory(
      dataFolder,
      seed=123,
      image_size=imageSize,
      batch_size=batchSize)
    return train_ds

# Tune buffer size and efficiency 
# When do we call this?
def configurePerformance(train_ds, val_ds): 
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

# def buildModel(dropoutRate, numClasses, inpShape = (224, 224, 3)):
import pickle

# Create pkl file of the model after the training phase
def dumpModel(modelName, phase): 
    # Save the trained model as a pickle string.
    modelName = "model_" + modelName + "_ " + phase + ".pkl"
    pickle.dump(model, open(modelName, 'wb'))
    
def getDatasetsByCar(cars,  imageSize = (224, 224), batchSize = 32):
    train_ds = None 
    val_ds = None 
    for car in cars: 
        trainingFolder = 'data/'+ car +'/train/RGB/'
        testingFolder = 'data/'+ car + '/test_with_labels/RGB/'
        if not train_ds and not val_ds:
            train_ds = getDataset(trainingFolder, "training")
            val_ds =  getDataset(testingFolder, "validation")
        else: 
            new_train_ds =  getDataset(trainingFolder, "training")
            new_val_ds =  getDataset(testingFolder, "validation")
            train_ds.concatenate(new_train_ds)
            val_ds.concatenate(new_val_ds)
    return train_ds, val_ds

EPOCHS = 5
modelName = "vgg16"
# Initial layer input shape
inpShape =  (224, 224, 3)
cars = ['x5', 'model3']
train_ds, val_ds = getDatasetsByCar(cars, batchSize=64)

# trainingFolder = 'data/x5/train/RGB/'
# testingFolder = 'data/x5/test_with_labels/RGB/'
# # don't need to pass subset string - datasets already split
# train_ds = getDataset(trainingFolder, "training")
# val_ds =  getDataset(testingFolder, "validation")

train_ds, val_ds = configurePerformance(train_ds, val_ds)

dropoutRate = 0.2
numClasses = 7 
inp = layers.Input(shape=inpShape)
baseModel = VGG16(weights="imagenet",
                   include_top = False) 
baseModel.trainable = False 
x = baseModel(inp, training=False)
x =  layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.Dropout(dropoutRate, noise_shape=None, seed=None)(x)
out = layers.Dense(numClasses,activation="softmax", name = "pred")(x)
model = keras.Model(inp, out, name="FeatureExtraction-B0")
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          # metrics=['accuracy']
          metrics=['accuracy',
              recall_m,
              precision_m,
              f1_m
              ]
          )

# Feature extraction without the top layers 
hist_results = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

dumpModel(modelName, "phase1")

# Fine tuning the Feature Extraction Model 
baseModel.trainable = True
for layer in model.layers[1].layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
        
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy',
              recall_m,
              precision_m,
              f1_m
              ]
              )

# Train it again 
hist_results_tuned = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=9,
  #steps_per_epoch=len(train_ds)?
  initial_epoch=hist_results.epoch[-1]
)

dumpModel(modelName, "phase2")

preds = model.predict(val_ds, verbose = 1)
model.evaluate(val_ds)
