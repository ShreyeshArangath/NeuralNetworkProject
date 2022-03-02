from tensorflow import keras 
from keras import layers
# from keras.preprocessing import image
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB5, EfficientNetB7
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.densenet import DenseNet121
# from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input

import os 
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential

# Data preprocessing 
from keras.preprocessing.image import load_img, img_to_array
 
# Load all the images from the folder 
dataFolder = os.path.dirname('data/x5_RGB/train/RGB/')
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

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
y = df['Labels'].values
yLabelEncoder = LabelEncoder()
y = yLabelEncoder.fit_transform(y)

from sklearn.compose import ColumnTransformer
y=y.reshape(-1,1)

ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y) #.toarray()
# print(Y[:5])
# print(Y[35:])

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
images, Y = shuffle(images,Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=1)

print('trainx shape: ', train_x.shape)
print('trainy shape: ', train_y.shape)
print('testx shape: ', test_x.shape)
print('testy shape: ', test_y.shape)

NUM_CLASSES = 7
IMG_SIZE = 224

# Set up the layers 
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
model = EfficientNetB0(include_top = False, input_tensor=inputs, weights="imagenet")
# efficientNetB5 = EfficientNetB5(weights="imagenet", include_top=False)  
# efficientNetB7 = EfficientNetB7(weights="imagenet", include_top=False)
# resNet152V2 = ResNet152V2(weights='imagenet', include_top=False)
# denseNet121 = DenseNet121(weights="imagenet")

# Freeze pretrained weights
model.trainable=False

# Rebuild top
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
# x = layers.GlobalAveragePooling2D(data_format=None, keepdims=False)(x)
# x = layers.BatchNormalization()(x)

RATE = 0.2
UNITS = 7

x = layers.Dropout(RATE, name='top_dropout')(x)
outputs = layers.Dense(UNITS, activation='softmax', name='Dense')(x)
# out = layers.Softmax(input_shape = (IMG_SIZE, IMG_SIZE, 3))(x)

# Compile
model = keras.Model(inputs=inputs, outputs=outputs, name='EfficientNet')
optimizer=keras.optimizers.Adam(learning_rate=1e-2)
model.compile(
    optimizer=optimizer, loss ='mse', metrics=['accuracy']
)
model.summary()

# Train the model 
hist = model.fit(train_x, train_y, epochs=5, verbose=2)

def plot_hist(hist):
    plt.plot(hist.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

plot_hist(hist)
# # Evaluate the model 

preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))