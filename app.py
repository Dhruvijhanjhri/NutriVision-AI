import numpy as np
import pandas as pd
import gradio as gr

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/mobilenet_food30_finetuned.keras"
NUTRITION_PATH = "data/nutrition/nutrition_database.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading NutriVision-AI model...")

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
# PORTION MULTIPLIERS
# ============================================================

portion_multiplier = {
    "Small": 0.75,
    "Medium": 1.00,
    "Large": 1.50
}

# ============================================================
# PERSONALIZATION
# ============================================================

def calculate_bmi(weight, height):
    """
    Calculate BMI from weight in kg and height in cm.
    """
    if height <= 0 or weight <= 0:
        return None

    height_m = height / 100

    return weight / (height_m ** 2)


def get_bmi_category(bmi):
    """
    Return a simple BMI category.
    """

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Normal weight"

    elif bmi < 30:
        return "Overweight"

    else:
        return "Obesity"


def calculate_daily_calories(
    age,
    gender,
    height,
    weight
):
    """
    Estimate daily calorie requirement using
    the Mifflin-St Jeor equation.

    Activity is kept at a moderate baseline for
    this project demonstration.
    """

    if age <= 0 or height <= 0 or weight <= 0:
        return None

    if gender == "Male":

        bmr = (
            10 * weight
            + 6.25 * height
            - 5 * age
            + 5
        )

    else:

        bmr = (
            10 * weight
            + 6.25 * height
            - 5 * age
            - 161
        )

    # Moderate baseline activity multiplier
    daily_calories = bmr * 1.55

    return daily_calories


def get_goal_message(
    goal,
    calories,
    daily_calories,
    health_score
):
    """
    Generate a simple personalized recommendation.
    """

    calorie_percentage = (
        calories / daily_calories
    ) * 100

    if goal == "Lose Weight":

        if calorie_percentage >= 25:

            message = (
                "This food contributes a relatively large "
                "portion of your estimated daily calories. "
                "Consider a smaller portion and balance it "
                "with vegetables and a protein-rich food."
            )

        else:

            message = (
                "This food can fit into a weight-management "
                "plan when consumed in an appropriate portion."
            )

    elif goal == "Gain Weight":

        message = (
            "This food can contribute useful calories toward "
            "your daily energy target. Consider pairing it "
            "with a protein-rich food for a more balanced meal."
        )

    else:

        message = (
            "This food can be included as part of a balanced "
            "diet while keeping your overall daily intake in mind."
        )

    if health_score <= 3:

        message += (
            " Its health score is relatively low, so moderation "
            "is recommended."
        )

    return message

# ============================================================
# FOOD PREDICTION
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
# PERSONALIZED RECOMMENDATION
# ============================================================

def generate_recommendation(row):

    health_score = float(row["HealthScore"])

    calories = float(row["Calories"])
    sugar = float(row["Sugar(g)"])
    sodium = float(row["Sodium(mg)"])

    recommendations = []

    if health_score >= 7:
        recommendations.append(
            "This food has a relatively good health score."
        )
    elif health_score >= 5:
        recommendations.append(
            "This food can be included in a balanced diet in moderation."
        )
    else:
        recommendations.append(
            "Consider consuming this food occasionally and in controlled portions."
        )

    if calories >= 400:
        recommendations.append(
            "Consider a smaller portion if you are monitoring calorie intake."
        )

    if sugar >= 20:
        recommendations.append(
            "This food is relatively high in sugar."
        )

    if sodium >= 500:
        recommendations.append(
            "This food is relatively high in sodium."
        )

    return " ".join(recommendations)


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_food(
    img_path,
    portion,
    age,
    gender,
    height,
    weight,
    goal
):

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if img_path is None:
        return (
            "### 🔎 Ready to Analyze\n\n"
            "Please upload a food image and click "
            "**Analyze Food**.",
            "",
            "",
            "",
            ""
        )

    # ========================================================
    # FOOD PREDICTION
    # ========================================================

    food, confidence, predictions = predict_food(img_path)

    # ========================================================
    # NUTRITION LOOKUP
    # ========================================================

    row = get_nutrition(food)

    if row is None:

        return (
            f"## 🤖 AI Food Prediction\n\n"
            f"### {food.replace('_', ' ').title()}\n\n"
            f"**AI Confidence:** {confidence:.2f}%\n\n"
            "Nutrition information was not found for this food.",
            "",
            "",
            "",
            ""
        )

    # ========================================================
    # PORTION ADJUSTMENT
    # ========================================================

    multiplier = portion_multiplier[portion]

    calories = float(row["Calories"]) * multiplier
    protein = float(row["Protein(g)"]) * multiplier
    carbs = float(row["Carbs(g)"]) * multiplier
    fat = float(row["Fat(g)"]) * multiplier
    fiber = float(row["Fiber(g)"]) * multiplier
    sugar = float(row["Sugar(g)"]) * multiplier
    sodium = float(row["Sodium(mg)"]) * multiplier

    health_score = float(row["HealthScore"])

    # ========================================================
    # TOP 5 PREDICTIONS
    # ========================================================

    top_indices = np.argsort(predictions)[::-1][:5]

    top_predictions = ""

    for rank, index in enumerate(top_indices, start=1):

        food_name = (
            class_names[index]
            .replace("_", " ")
            .title()
        )

        probability = predictions[index] * 100

        top_predictions += (
            f"{rank}. **{food_name}** — "
            f"{probability:.2f}%\n\n"
        )

    # ========================================================
    # PERSONALIZATION
    # ========================================================

    bmi = None
    bmi_category = "Not calculated"
    daily_calories = None
    calorie_percentage = None

    recommendation = (
        "Enter valid profile information to receive "
        "personalized nutrition insights."
    )

    if (
        age is not None
        and height is not None
        and weight is not None
        and age > 0
        and height > 0
        and weight > 0
    ):

        bmi = calculate_bmi(
            weight,
            height
        )

        bmi_category = get_bmi_category(bmi)

        daily_calories = calculate_daily_calories(
            age,
            gender,
            height,
            weight
        )

        calorie_percentage = (
            calories / daily_calories
        ) * 100

        recommendation = get_goal_message(
            goal,
            calories,
            daily_calories,
            health_score
        )

    # ========================================================
    # PREDICTION OUTPUT
    # ========================================================

    prediction_text = (
        "## 🤖 AI Food Prediction\n\n"
        f"### {food.replace('_', ' ').title()}\n\n"
        f"**AI Confidence:** {confidence:.2f}%\n\n"
        f"**Portion:** {portion}\n\n"
        f"### ⭐ Health Score: {health_score:.0f}/10"
    )

    # ========================================================
    # NUTRITION OUTPUT
    # ========================================================

    nutrition_text = (
        "## 🥗 Nutrition Analysis\n\n"
        f"### 🔥 {calories:.2f} kcal\n\n"
        f"**Protein:** {protein:.2f} g  \n"
        f"**Carbohydrates:** {carbs:.2f} g  \n"
        f"**Fat:** {fat:.2f} g  \n"
        f"**Fiber:** {fiber:.2f} g  \n"
        f"**Sugar:** {sugar:.2f} g  \n"
        f"**Sodium:** {sodium:.2f} mg"
    )

    # ========================================================
    # TOP 5 OUTPUT
    # ========================================================

    top5_text = (
        "## 🔍 Top 5 AI Predictions\n\n"
        + top_predictions
    )

    # ========================================================
    # FOOD INFORMATION
    # ========================================================

    description_text = (
        "## 📖 Food Information\n\n"
        f"{row['Description']}"
    )

    # ========================================================
    # PERSONALIZED OUTPUT
    # ========================================================

    if daily_calories is not None:

        personalized_text = (
            "## 🎯 Personalized Nutrition\n\n"
            f"**BMI:** {bmi:.1f}\n\n"
            f"**BMI Category:** {bmi_category}\n\n"
            f"**Estimated Daily Calories:** "
            f"{daily_calories:.0f} kcal\n\n"
            f"**This Food:** {calorie_percentage:.1f}% "
            f"of estimated daily calories\n\n"
            f"### 💡 Recommendation\n\n"
            f"{recommendation}"
        )

    else:

        personalized_text = (
            "## 🎯 Personalized Nutrition\n\n"
            "Enter your age, gender, height and weight "
            "to calculate BMI and estimated daily "
            "calorie requirements.\n\n"
            f"### 💡 Recommendation\n\n"
            f"{recommendation}"
        )

    # ========================================================
    # HEALTH OUTPUT
    # ========================================================

    health_text = (
        "## ⭐ Health Score\n\n"
        f"# {health_score:.0f}/10"
    )

    # ========================================================
    # RETURN EXACTLY 6 OUTPUTS
    # ========================================================

    return (
        prediction_text,
        top5_text,
        nutrition_text,
        description_text,
        personalized_text,
        health_text
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate",
)

with gr.Blocks(
    title="NutriVision-AI",
    theme=theme,
    css="""
    .main-header {
        text-align: center;
        padding: 20px 10px;
    }

    .main-header h1 {
        font-size: 42px;
        font-weight: 700;
    }

    .main-header p {
        font-size: 17px;
        color: #64748b;
    }

    .section-title {
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .analyze-btn {
        margin-top: 10px;
    }

    /* Result Cards */

    .result-card {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        background: white;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
    }

    .result-card h2 {
        margin-top: 0;
    }

    /* Profile Card */

    .profile-card {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
    }

    .section-header {
        margin-top: 10px;
        margin-bottom: 5px;
    }
    """
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown(
        """
        <div class="main-header">

        # 🍽️ NutriVision-AI

        ### AI-Powered Food Recognition & Personalized Nutrition

        Identify food from an image, estimate nutrition,
        evaluate health score, and receive personalized dietary insights.

        </div>
        """
    )

    gr.Markdown("---")

    # ========================================================
    # USER PROFILE
    # ========================================================

    gr.Markdown(
        """
        ## 👤 Your Profile

        Enter your basic information to receive personalized
        nutrition insights.
        """
    )

    with gr.Row():

        age_input = gr.Number(
            label="Age",
            value=23,
            minimum=1,
            maximum=100,
            precision=0
        )

        gender_input = gr.Dropdown(
            choices=["Female", "Male"],
            value="Female",
            label="Gender"
        )

    with gr.Row():

        height_input = gr.Number(
            label="Height (cm)",
            value=165,
            minimum=50,
            maximum=250
        )

        weight_input = gr.Number(
            label="Weight (kg)",
            value=60,
            minimum=10,
            maximum=300
        )

        goal_input = gr.Dropdown(
            choices=[
                "Maintain Weight",
                "Lose Weight",
                "Gain Weight"
            ],
            value="Maintain Weight",
            label="Health Goal"
        )

    gr.Markdown("---")

    # ========================================================
    # FOOD IMAGE ANALYSIS
    # ========================================================

    gr.Markdown(
        """
        ## 📸 Analyze Your Food

        Upload an image of a food item and let the AI model
        identify it.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="filepath",
                label="Upload Food Image"
            )

            portion_input = gr.Radio(
                choices=[
                    "Small",
                    "Medium",
                    "Large"
                ],
                value="Medium",
                label="Estimated Portion"
            )

            analyze_button = gr.Button(
                "🔍 Analyze My Food",
                variant="primary",
                size="lg",
                elem_classes="analyze-btn"
            )

        with gr.Column():

            prediction_output = gr.Markdown(
                """
                ### 🔎 Ready to Analyze

                Upload a food image and click
                **Analyze Food**.
                """
            )

    gr.Markdown("---")

    # ========================================================
    # NUTRITION INFORMATION
    # ========================================================

    with gr.Group(elem_classes="result-card"):

        nutrition_output = gr.Markdown(
            ""
        )

    # ========================================================
    # AI RESULTS
    # ========================================================

    with gr.Row():

        with gr.Column(elem_classes="result-card"):

            top5_output = gr.Markdown(
                ""
            )

        # with gr.Column(elem_classes="result-card"):

        #     nutrition_output = gr.Markdown(
        #         ""
        #     )
            
    # ========================================================
    # FOOD INFORMATION + HEALTH SCORE
    # ========================================================

    with gr.Row():

        with gr.Column(elem_classes="result-card"):

            description_output = gr.Markdown("")

        with gr.Column(elem_classes="result-card"):

            health_output_2 = gr.Markdown("")

    # ========================================================
    # PERSONALIZED NUTRITION
    # ========================================================

    personalized_output = gr.Markdown("")

    gr.Markdown("---")

    # ========================================================
    # HOW IT WORKS
    # ========================================================

    gr.Markdown(
        """
        ### 🧠 How NutriVision-AI Works

        **Food Image → MobileNetV2 → Food Classification →
        Confidence → Nutrition Knowledge Base →
        Portion Adjustment → Personalized Nutrition Insights**

        **Model:** MobileNetV2 Fine-Tuned on 30 Food Classes  
        **Framework:** TensorFlow / Keras  
        **Interface:** Gradio
        """
    )

    # ========================================================
    # BUTTON ACTION
    # ========================================================

    analyze_button.click(
        fn=analyze_food,

        inputs=[
            image_input,
            portion_input,
            age_input,
            gender_input,
            height_input,
            weight_input,
            goal_input
        ],

        outputs=[
            prediction_output,
            top5_output,
            nutrition_output,
            description_output,
            personalized_output,
            health_output_2
        ]
    )


# ============================================================
# LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch()