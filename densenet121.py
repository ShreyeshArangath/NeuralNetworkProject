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
modelName = "denseNet121"
inpShape =  (224, 224, 3)
trainingFolder = 'data/x5/train/RGB/'
testingFolder = 'data/x5/test_with_labels/RGB/'
train_ds = getDataset(trainingFolder, "training")
val_ds =  getDataset(testingFolder, "validation")

dropoutRate = 0.2
numClasses = 7 
inp = layers.Input(shape=inpShape)
baseModel = DenseNet121(weights="imagenet",
                   include_top = False) 

baseModel.trainable = False 
x = baseModel(inp, training=False)
x =  layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.Dropout(dropoutRate, noise_shape=None, seed=None)(x)
out = layers.Dense(numClasses,activation="softmax", name = "pred")(x)
model = keras.Model(inp, out, name="FeatureExtraction-denseNet121")
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

# Fine tuning the Feature Extraction Model 
baseModel.trainable = True
for layer in model.layers[1].layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
        
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics = ["accuracy"])

# Train it again 
hist_results_tuned = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

dumpModel(modelName, "phase2")

preds = model.predict(val_ds, verbose = 1)
model.evaluate(val_ds)

# def getImageRepresentation():
#     import random
#     pred_labels = tf.argmax(preds, axis=1)
#     test_labels = np.concatenate([y for x, y in val_ds], axis=0)
#     test_image_batches = []
#     for images, labels in val_ds.take(-1):
#         test_image_batches.append(images.numpy())

#     test_images = [item for sublist in test_image_batches for item in sublist]
#     plt.figure(figsize = (20,20))
#     for i in range(9):
#         random_int_index = random.choice(range(len(test_images)))
#         plt.subplot(3,3,i+1)
#         plt.imshow(test_images[random_int_index]/255.)
#         if test_labels[random_int_index] == pred_labels[random_int_index]:
#             color = "g"
#         else:
#             color = "r"
#         plt.title("True Label: " + class_names[test_labels[random_int_index]] + " || " + "Predicted Label: " +
#                   class_names[pred_labels[random_int_index]] + "\n" + 
#                   str(np.asarray(tf.reduce_max(preds, axis = 1))[random_int_index]), c=color)
#         plt.axis(False);