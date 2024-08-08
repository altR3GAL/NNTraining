import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, regularizers
import pickle
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-100 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar100.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Function to load class names from CIFAR-100 meta file
def load_class_names():
    with open('cifar-100-python/cifar-100-python/meta', 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        return [name.decode('utf-8') for name in meta[b'fine_label_names']]

# Load CIFAR-100 class names
class_names = load_class_names()

# Display the first 16 images with their labels
# for i in range(16):
#     plt.subplot(4, 4, i + 1)  # Create a subplot with 4x4 grid
#     plt.xticks([])  # Remove x-axis ticks
#     plt.yticks([])  # Remove y-axis ticks
#     plt.imshow(training_images[i], cmap=plt.cm.binary)  # Display image
#     plt.xlabel(class_names[training_labels[i][0]])  # Set the label as the class name

# Optionally, limit the dataset size for faster training during development
# training_images = training_images[:20000]
# training_labels = training_labels[:20000]
# testing_images = testing_images[:5000]
# testing_labels = testing_labels[:5000]

# training_images = training_images[:20000]
# training_labels = training_labels[:20000]
# testing_images = testing_images[:5000]
# testing_labels = testing_labels[:5000]

# Define an enhanced model architecture with dropout and batch normalization
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(100, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Prepare the data augmentation on the training data
datagen.fit(training_images)

# Train the model with data augmentation
history = model.fit(datagen.flow(training_images, training_labels, batch_size=64), epochs=50, validation_data=(testing_images, testing_labels))

# Save the trained model
model.save('enhanced_image_classifier.keras')
