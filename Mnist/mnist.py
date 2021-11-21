import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
model = load_model("mnist.h5")
model.summary()
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
folder = "numbers"
images = []
names = []
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename), 0)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = round((100 * np.max(predictions)), 2)
    name = class_names[np.argmax(predictions)]
    names.append(name)
    if img is not None:
        images.append(img)
fig = plt.figure()
for i in range(len(images)):
    j = i + 1
    fig.add_subplot(2, 5, j)
    plt.xlabel(names[i])
    plt.imshow(images[i])

plt.show()
