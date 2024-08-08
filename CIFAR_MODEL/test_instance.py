import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import pickle
import os

# Function to load class names from CIFAR-100 meta file
def load_class_names():
    with open('cifar-100-python/meta', 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        return [name.decode('utf-8') for name in meta[b'fine_label_names']]

# Load CIFAR-100 class names
class_names = load_class_names()

# Load the pre-trained model
model = models.load_model('image_classifier.keras')

img = cv.imread('images/panda.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))

plt.imshow(img, cmap = plt.cm.binary)
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Predicition is {class_names[index]}")