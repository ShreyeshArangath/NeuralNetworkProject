from tensorflow import keras 
from keras import layers, applications
import os 

# Data preprocessing 
from keras.preprocessing.image import load_img

# Load all the images from the folder 
dataFolder = os.path.dirname('data/x5_RGB/train/RGB/')
imagePaths = []

for case in os.listdir(dataFolder):
    print(case)
    f = os.path.join(dataFolder, case)
    for image in os.listdir(f):
        imagePath = os.path.join(dataFolder, case, image)
        imagePaths.append(imagePath)

print(imagePaths)


# # Set up the layers 
# efficientNetB0 = applications.EfficientNetB0(
#     include_top=True,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax"
# )

# efficientNetB5 = applications.EfficientNetB5(
#     include_top=True,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax"
# )

# efficientNetB7 = applications.EfficientNetB7(
#     include_top=True,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax"
# )

#  # ResNet152V2 

# # DenseNet121

# globalAveragePooling = layers.GlobalAveragePooling2D(
#     data_format=None, keepdims=False
# )

# dropout = layers.Dropout(rate, noise_shape=None, seed=None)

# dense = layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer="glorot_uniform",
#     bias_initializer="zeros",
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None
# )

# softmax = layers.Softmax()

# Train the model 

# Evaluate the model 