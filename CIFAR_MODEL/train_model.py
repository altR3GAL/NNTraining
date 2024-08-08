import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D
import pickle
import os

# Load the CIFAR-100 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar100.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Function to load class names from CIFAR-100 meta file
def load_class_names():
    with open('cifar-100-python/meta', 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        return [name.decode('utf-8') for name in meta[b'fine_label_names']]

# Load CIFAR-100 class names
class_names = load_class_names()

# Define an enhanced model architecture with dropout, batch normalization, and regularization
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

    GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(100, activation='softmax')
])

# Use SGD with momentum for better generalization
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# Compile the model with the optimizer and loss function
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,  # Adding zoom augmentation
    shear_range=0.2  # Adding shear augmentation
)

# Prepare the data augmentation on the training data
datagen.fit(training_images)

# Learning rate scheduler and early stopping to prevent overfitting
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with data augmentation and callbacks
history = model.fit(
    datagen.flow(training_images, training_labels, batch_size=64), 
    epochs=50, 
    validation_data=(testing_images, testing_labels),
    callbacks=[lr_reduction, early_stopping]
)

# Save the trained model
model.save('enhanced_image_classifier.keras')

# Optionally, display the first 16 images with their labels
# for i in range(16):
#     plt.subplot(4, 4, i + 1)  # Create a subplot with 4x4 grid
#     plt.xticks([])  # Remove x-axis ticks
#     plt.yticks([])  # Remove y-axis ticks
#     plt.imshow(training_images[i], cmap=plt.cm.binary)  # Display image
#     plt.xlabel(class_names[training_labels[i][0]])  # Set the label as the class name
