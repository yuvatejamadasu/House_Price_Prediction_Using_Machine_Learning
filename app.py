"""
===============================================================
        VIHARATECH MACHINE LEARNING PROJECT
                  HOUSE PRICE PREDICTION

        Algorithm: Multiple Linear Regression

        Developed by: M. Yuva Teja
===============================================================
"""

# =============================================================
# IMPORT LIBRARIES
# =============================================================

import os
import pickle
import logging

import numpy as np

from flask import Flask, render_template, request


# =============================================================
# FLASK APPLICATION CONFIGURATION
# =============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "house-price-prediction-secret"
)

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "House_Price_Prediction_Model.pkl"
)


# =============================================================
# MODEL CONFIGURATION
# =============================================================

MODEL = None

FEATURE_NAMES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "country",
    "state",
    "zipcode"
]


# =============================================================
# LOAD MACHINE LEARNING MODEL
# =============================================================

def load_model():

    global MODEL

    try:

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        with open(MODEL_PATH, "rb") as file:

            MODEL = pickle.load(file)

        # Verify model feature count

        expected_features = getattr(
            MODEL,
            "n_features_in_",
            len(FEATURE_NAMES)
        )

        if expected_features != len(FEATURE_NAMES):

            raise ValueError(
                f"Model expects {expected_features} features, "
                f"but application supplies {len(FEATURE_NAMES)}."
            )

        app.logger.info(
            "House Price Prediction Model loaded successfully."
        )

        return True

    except Exception as error:

        app.logger.exception(
            "Model loading failed: %s",
            error
        )

        return False


# Load model when application starts

load_model()


# =============================================================
# DEFAULT PROPERTY VALUES
# =============================================================

DEFAULT_VALUES = {

    "bedrooms": 4,
    "bathrooms": 2,
    "sqft_living": 2609,
    "sqft_lot": 5000,
    "floors": 1,
    "waterfront": 0,
    "view": 2,
    "condition": 4,
    "sqft_above": 1509,
    "sqft_basement": 1100,
    "yr_built": 2004,
    "yr_renovated": 2020,
    "zipcode": 98001

}


# =============================================================
# INPUT VALIDATION
# =============================================================

def validate_input(data):

    errors = []

    try:

        bedrooms = int(data["bedrooms"])
        bathrooms = float(data["bathrooms"])

        sqft_living = float(data["sqft_living"])
        sqft_lot = float(data["sqft_lot"])

        floors = float(data["floors"])

        waterfront = int(data["waterfront"])
        view = int(data["view"])
        condition = int(data["condition"])

        sqft_above = float(data["sqft_above"])
        sqft_basement = float(data["sqft_basement"])

        yr_built = int(data["yr_built"])
        yr_renovated = int(data["yr_renovated"])

        zipcode = int(data["zipcode"])

        # -----------------------------------------------------
        # BASIC VALIDATION
        # -----------------------------------------------------

        if bedrooms < 1 or bedrooms > 20:
            errors.append("Bedrooms must be between 1 and 20.")

        if bathrooms < 0 or bathrooms > 20:
            errors.append("Invalid bathroom value.")

        if sqft_living <= 0 or sqft_living > 100000:
            errors.append("Invalid living area.")

        if sqft_lot <= 0 or sqft_lot > 10000000:
            errors.append("Invalid lot area.")

        if floors <= 0 or floors > 10:
            errors.append("Invalid floor value.")

        if waterfront not in [0, 1]:
            errors.append("Waterfront must be 0 or 1.")

        if view not in range(5):
            errors.append("View must be between 0 and 4.")

        if condition not in range(1, 6):
            errors.append("Condition must be between 1 and 5.")

        if sqft_above < 0 or sqft_above > 100000:
            errors.append("Invalid above-ground area.")

        if sqft_basement < 0 or sqft_basement > 100000:
            errors.append("Invalid basement area.")

        if yr_built < 1800 or yr_built > 2026:
            errors.append("Invalid construction year.")

        if yr_renovated != 0:

            if yr_renovated < yr_built:
                errors.append(
                    "Renovation year cannot be earlier than construction year."
                )

            if yr_renovated > 2026:
                errors.append("Invalid renovation year.")

        if zipcode < 10000 or zipcode > 99999:
            errors.append("ZIP code must contain 5 digits.")

        if not errors:

            return {

                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft_living": sqft_living,
                "sqft_lot": sqft_lot,
                "floors": floors,
                "waterfront": waterfront,
                "view": view,
                "condition": condition,
                "sqft_above": sqft_above,
                "sqft_basement": sqft_basement,
                "yr_built": yr_built,
                "yr_renovated": yr_renovated,
                "country": 0,
                "state": 0,
                "zipcode": zipcode

            }

    except (ValueError, TypeError, KeyError):

        errors.append(
            "Please enter valid numerical values in all fields."
        )

    return errors


# =============================================================
# HOUSE PRICE PREDICTION FUNCTION
# =============================================================

def predict_house_price(property_data):

    try:

        if MODEL is None:

            raise RuntimeError(
                "Machine Learning model is not loaded."
            )

        # -----------------------------------------------------
        # FEATURE ORDER MUST MATCH main.py
        # -----------------------------------------------------

        input_values = [

            property_data["bedrooms"],
            property_data["bathrooms"],
            property_data["sqft_living"],
            property_data["sqft_lot"],
            property_data["floors"],
            property_data["waterfront"],
            property_data["view"],
            property_data["condition"],
            property_data["sqft_above"],
            property_data["sqft_basement"],
            property_data["yr_built"],
            property_data["yr_renovated"],
            property_data["country"],
            property_data["state"],
            property_data["zipcode"]

        ]

        input_array = np.array(
            [input_values],
            dtype=float
        )

        # -----------------------------------------------------
        # MODEL PREDICTION
        # -----------------------------------------------------

        prediction = MODEL.predict(input_array)

        predicted_price = float(prediction[0])

        if not np.isfinite(predicted_price):
            raise ValueError("The model returned an invalid prediction.")

        if predicted_price < 0:
            raise ValueError(
                "The model returned a negative price. "
                "Please verify the input values and training data."
            )

        return predicted_price

    except Exception as error:

        app.logger.exception(
            "Prediction failed: %s",
            error
        )

        raise


# =============================================================
# MAIN APPLICATION ROUTE
# =============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    error = None

    form_data = DEFAULT_VALUES.copy()

    if request.method == "POST":

        form_data = request.form.to_dict()

        validation_result = validate_input(form_data)

        if isinstance(validation_result, list):

            error = " ".join(validation_result)

        else:

            try:

                prediction = predict_house_price(
                    validation_result
                )

                app.logger.info(
                    "Prediction generated: %.2f",
                    prediction
                )

            except Exception as exception:

                error = str(exception)

    return render_template(

        "index.html",

        prediction=prediction,

        error=error,

        form_data=form_data,

        default_values=DEFAULT_VALUES

    )


# =============================================================
# APPLICATION HEALTH CHECK
# =============================================================

@app.route("/health")
def health():

    return {

        "application": "House Price Prediction",

        "algorithm": "Multiple Linear Regression",

        "model_status": (
            "Loaded" if MODEL is not None else "Not Loaded"
        )

    }


# =============================================================
# RUN FLASK APPLICATION
# =============================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )