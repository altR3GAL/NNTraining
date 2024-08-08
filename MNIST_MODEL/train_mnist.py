# Updated `train_model.py` script using the MNIST dataset

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D

# Load the MNIST dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.mnist.load_data()

# Reshape the dataset to include the channel dimension and normalize
training_images = training_images.reshape((training_images.shape[0], 28, 28, 1)) / 255.0
testing_images = testing_images.reshape((testing_images.shape[0], 28, 28, 1)) / 255.0

# Define an enhanced model architecture with dropout, batch normalization, and regularization
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    GlobalAveragePooling2D(),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')  # MNIST has 10 classes
])

# Use SGD with momentum for better generalization
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# Compile the model with the optimizer and loss function
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation (optional for MNIST since it's grayscale)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)

# Prepare the data augmentation on the training data
datagen.fit(training_images)

# Learning rate scheduler and early stopping to prevent overfitting
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with data augmentation and callbacks
history = model.fit(
    datagen.flow(training_images, training_labels, batch_size=64), 
    epochs=10, 
    validation_data=(testing_images, testing_labels),
    callbacks=[lr_reduction, early_stopping]
)

# Save the trained model
model.save('mnist_model_10epoch.keras')  # Updated the save path and model name

# Optionally, display the first 16 images with their labels
# for i in range(16):
#     plt.subplot(4, 4, i + 1)  # Create a subplot with 4x4 grid
#     plt.xticks([])  # Remove x-axis ticks
#     plt.yticks([])  # Remove y-axis ticks
#     plt.imshow(training_images[i].reshape(28, 28), cmap=plt.cm.binary)  # Display image
#     plt.xlabel(training_labels[i])  # Set the label as the class name
