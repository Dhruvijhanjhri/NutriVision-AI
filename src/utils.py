import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# ---------------------------------------------------
# Create Folder
# ---------------------------------------------------

def create_directory(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------
# Save Class Names
# ---------------------------------------------------

def save_class_names(class_names, filepath):

    with open(filepath, "w") as f:
        json.dump(class_names, f)

    print("Class names saved.")


# ---------------------------------------------------
# Load Class Names
# ---------------------------------------------------

def load_class_names(filepath):

    with open(filepath, "r") as f:
        class_names = json.load(f)

    return class_names


# ---------------------------------------------------
# Save Training History
# ---------------------------------------------------

def save_history(history, filepath):

    with open(filepath, "w") as f:
        json.dump(history.history, f)


# ---------------------------------------------------
# Plot Accuracy
# ---------------------------------------------------

def plot_accuracy(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="Train")

    plt.plot(history.history["val_accuracy"], label="Validation")

    plt.title("Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.show()


# ---------------------------------------------------
# Plot Loss
# ---------------------------------------------------

def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"], label="Train")

    plt.plot(history.history["val_loss"], label="Validation")

    plt.title("Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.show()


# ---------------------------------------------------
# Load Saved Model
# ---------------------------------------------------

def load_saved_model(model_path):

    return load_model(model_path)


# ---------------------------------------------------
# Predict Single Image
# ---------------------------------------------------

def predict_image(model, image, class_names):

    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    return class_names[index], confidence