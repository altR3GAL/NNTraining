import os
import cv2 as cv
import numpy as np
from tensorflow.keras import models
import pickle

# Function to load class names from CIFAR-100 meta file
def load_class_names():
    with open('cifar-100-python/meta', 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        return [name.decode('utf-8') for name in meta[b'fine_label_names']]

# Load CIFAR-100 class names
class_names = load_class_names()

# Load the pre-trained model
model = models.load_model('image_classifier.keras')

# Function to predict the class of an image
def predict_image_class(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (32, 32))
    prediction = model.predict(np.array([img]) / 255.0)
    index = np.argmax(prediction)
    return class_names[index]

# Function to test all images in the folder
def test_images(folder_path):
    incorrect_images = []
    
    for image_file in os.listdir(folder_path):
        # Ensure it's an image file
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, image_file)
            predicted_class = predict_image_class(image_path)
            actual_class = os.path.splitext(image_file)[0].lower()  # Get the name without the extension

            if predicted_class.lower() != actual_class:
                incorrect_images.append(image_file)

    return incorrect_images

# Main function to run the tests
if __name__ == "__main__":
    folder_path = 'images'
    incorrect_images = test_images(folder_path)
    
    if len(incorrect_images) == 0:
        print("Pass: All images were correctly identified.")
    else:
        print("Fail: The following images were not correctly identified:")
        for image in incorrect_images:
            print(image)
