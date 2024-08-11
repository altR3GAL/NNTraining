import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

# Class names for binary classification
class_names = ['sad', 'happy']

# Load the pre-trained model
model = models.load_model('models/scratch_model.keras')

# Load and preprocess the image
img = cv.imread('data/happy/05-12-21-happy-people.jpg')
img = cv.resize(img, (256, 256))  # Resize to the model's expected input size
img = img / 255.0  # Normalize the image
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make a prediction
prediction = model.predict(img)
prediction_label = class_names[int(prediction < 0.5)]  # Binary classification logic

print(f"Prediction is: {prediction_label}")
