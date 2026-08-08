import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/mobilenet_food30_finetuned.keras"
NUTRITION_PATH = "data/nutrition/nutrition_database.csv"


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "apple_pie",
    "beef_carpaccio",
    "caesar_salad",
    "cheesecake",
    "chicken_curry",
    "chicken_wings",
    "chocolate_cake",
    "club_sandwich",
    "dumplings",
    "fish_and_chips",
    "french_fries",
    "fried_calamari",
    "fried_rice",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "lasagna",
    "macaroni_and_cheese",
    "omelette",
    "pho",
    "pizza",
    "ramen",
    "samosa",
    "spaghetti_bolognese",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "waffles"
]


# ============================================================
# LOAD NUTRITION DATABASE
# ============================================================

nutrition = pd.read_csv(NUTRITION_PATH)

print("Nutrition database loaded successfully!")


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_food(img_path):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = predictions[predicted_index] * 100

    return predicted_class, confidence, predictions


# ============================================================
# NUTRITION LOOKUP
# ============================================================

def get_nutrition(food_name):

    row = nutrition[
        nutrition["Food"].str.lower() == food_name.lower()
    ]

    if row.empty:
        return None

    return row.iloc[0]


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = input("Enter image path: ").strip()

food, confidence, predictions = predict_food(IMAGE_PATH)

row = get_nutrition(food)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("           LOCAL PIPELINE TEST")
print("=" * 60)

print(f"Food       : {food}")
print(f"Confidence : {confidence:.2f}%")


if row is not None:

    print()
    print("Nutrition information found!")

    print(f"Calories   : {row['Calories']} kcal")
    print(f"Protein    : {row['Protein(g)']} g")
    print(f"Carbs      : {row['Carbs(g)']} g")
    print(f"Fat        : {row['Fat(g)']} g")
    print(f"Fiber      : {row['Fiber(g)']} g")
    print(f"Sugar      : {row['Sugar(g)']} g")
    print(f"Sodium     : {row['Sodium(mg)']} mg")
    print(f"HealthScore: {row['HealthScore']}")

else:

    print("Nutrition information not found.")

print("=" * 60)