"""
model_builder.py

Builds transfer learning models for NutriVision-AI.
Supports:
1. MobileNetV2
2. ResNet50
3. EfficientNetB0
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from .config import (
    IMAGE_SIZE,
    DENSE_UNITS,
    DROPOUT_RATE_1,
    DROPOUT_RATE_2,
)

# ============================================================
# Common Classification Head
# ============================================================

def build_classifier_head(base_model, preprocessing_function, num_classes):

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    x = preprocessing_function(inputs)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(DROPOUT_RATE_1)(x)

    x = layers.Dense(
        DENSE_UNITS,
        activation="relu"
    )(x)

    x = layers.Dropout(DROPOUT_RATE_2)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    return model


# ============================================================
# MobileNetV2
# ============================================================

def build_mobilenet(num_classes):

    base_model = tf.keras.applications.MobileNetV2(

        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

        include_top=False,

        weights="imagenet"

    )

    base_model.trainable = False

    model = build_classifier_head(

        base_model,

        tf.keras.applications.mobilenet_v2.preprocess_input,

        num_classes

    )

    return model, base_model


# ============================================================
# ResNet50
# ============================================================

def build_resnet50(num_classes):

    base_model = tf.keras.applications.ResNet50(

        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

        include_top=False,

        weights="imagenet"

    )

    base_model.trainable = False

    model = build_classifier_head(

        base_model,

        tf.keras.applications.resnet.preprocess_input,

        num_classes

    )

    return model, base_model


# ============================================================
# EfficientNetB0
# ============================================================

def build_efficientnet(num_classes):

    base_model = tf.keras.applications.EfficientNetB0(

        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

        include_top=False,

        weights="imagenet"

    )

    base_model.trainable = False

    model = build_classifier_head(

        base_model,

        tf.keras.applications.efficientnet.preprocess_input,

        num_classes

    )

    return model, base_model