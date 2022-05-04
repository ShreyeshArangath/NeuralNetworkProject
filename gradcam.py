from tabnanny import verbose
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from keras import models, layers
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_image(img_path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads  = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcame(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap('jet')

    jet_colors = jet(np.arange(265))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

# redefine metrics for custom_objects (load_model)
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

def configurePerformance(train_ds, val_ds): 
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

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

def getDataset(dataFolder, subset, imageSize = (224, 224), batchSize = 32):
    train_ds = keras.utils.image_dataset_from_directory(
      dataFolder,
      seed=123,
      image_size=imageSize,
      batch_size=batchSize)
    return train_ds


EPOCHS = 5
modelName = "vgg16"
inpShape =  (224, 224, 3)
cars = ['x5', 'model3']
train_ds, val_ds = getDatasetsByCar(cars, batchSize=64)

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
baseModel = VGG16(weights='imagenet')

augmentation = tf.keras.Sequential([
  # Add the preprocessing layers you created earlier.
  resize_and_rescale,
  data_augmentation,
])

inp = augmentation(inp)


baseModel.trainable = False 
x = baseModel(inp, training=False)
print(baseModel.summary())
x = layers.Dropout(dropoutRate, noise_shape=None, seed=None)(x)
out = layers.Dense(numClasses,activation="softmax", name = "pred")(x)
model = keras.Model(inp, out, name="FeatureExtraction-B0")

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

)

input = load_image(sys.argv[1])
print(model.summary())
preds = model.predict(val_ds, verbose=1)
print(preds)
top_1 = np.argmax(preds)
print("Predicted:", top_1)
heatmap = make_gradcam_heatmap(input, baseModel, last_conv_layer_name="block5_conv3")
save_and_display_gradcame(img_path=sys.argv[1], heatmap=heatmap)



