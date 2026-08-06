"""
data_loader.py

Loads and prepares the Food-30 dataset for training.
"""

import tensorflow as tf
from tensorflow.keras import layers

from .config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
)

# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)

# ============================================================
# Load Dataset
# ============================================================

def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False,
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
        .prefetch(AUTOTUNE)
    )

    val_ds = val_ds.prefetch(AUTOTUNE)

    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

# ============================================================
# Get Data Augmentation
# ============================================================

def get_data_augmentation():
    return data_augmentation