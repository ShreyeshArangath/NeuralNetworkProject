from email.utils import decode_rfc2231
from tabnanny import verbose
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from keras import models, layers, Sequential
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
        class_channel = preds[:,1]

    grads  = tape.gradient(class_channel, preds)

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

classifications = ['empty', 'Infant in infant seat', 'Child in child seat', 'Adult', 'Everyday object', 'Empty infant seat', 'Empty child seat']
model = keras.models.load_model('t_vgg_phase2.h5', custom_objects={"precision_m": precision_m, "f1_m": f1_m, "recall_m": recall_m})
#model = VGG16(weights="imagenet", include_top=True)

input = load_image(sys.argv[1])
vgg_model = model.layers[1]
preds = model.predict(input, verbose=1)
top_1 = np.argmax(preds)
print(preds)
print("Predicted:", classifications[top_1])
print("Prediction Sureness: ", preds[0][top_1])

heatmap = make_gradcam_heatmap(input, vgg_model, last_conv_layer_name="block5_conv3") #block5_conv3
save_and_display_gradcame(img_path=sys.argv[1], heatmap=heatmap)


