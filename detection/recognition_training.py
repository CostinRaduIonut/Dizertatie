import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten,
    Input, Rescaling, Dropout, BatchNormalization, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

datagen = ImageDataGenerator(rotation_range=20,
                             shear_range=10,
                             validation_split=0.2)

# Path to dataset (adjust as needed)
base_dir = "detection/recognition_dataset/"
image_shape = (28, 28, 3)
image_size = image_shape[:2]
BATCH_SIZE = 16

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=(28, 28),
    subset='training'
)

valid_data = datagen.flow_from_directory(
    base_dir,
    target_size=(28, 28),
    subset='validation'
)

# CNN Model
def get_model():
    inputs = Input(shape=image_shape)
    x = Rescaling(1. / 255)(inputs)

    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(26, activation="softmax")(x)
    return Model(inputs, outputs)

# Build and compile model
model = get_model()
model.summary()
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=100
)

# Save model
model.save("nn_braille")

# Plot results
epochs = range(1, len(history.history["accuracy"]) + 1)

plt.plot(epochs, history.history["accuracy"], "bo", label="Training Accuracy")
plt.plot(epochs, history.history["val_accuracy"], "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, history.history["loss"], "ro", label="Training Loss")
plt.plot(epochs, history.history["val_loss"], "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
