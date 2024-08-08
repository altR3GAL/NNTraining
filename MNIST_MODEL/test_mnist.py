import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# MNIST class names
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Load the pre-trained model
model = models.load_model('mnist_model.keras')

# Load and preprocess the image
img = cv.imread('numbers/four.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (28, 28))

plt.imshow(img, cmap='gray')
prediction = model.predict(np.array([img]).reshape(1, 28, 28, 1) / 255.0)
index = np.argmax(prediction)
print(f"Prediction is {class_names[index]}")
