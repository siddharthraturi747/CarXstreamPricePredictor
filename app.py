from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

MODEL_PATH = "model/car_price_predictor.pkl"
ENCODER_PATH = "model/encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✅ Model and encoder loaded successfully.")
    print("Model Expected Features:", model.feature_names_in_)
except Exception as e:
    print(f"❌ Error loading model or encoder: {e}")
    model = None
    encoder = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or encoder is None:
        return render_template('index.html', error="❌ Model or encoder not loaded.")

    try:
        # 1. Retrieve input values from the form (THIS WAS MISSING)
        kilometers = request.form.get('Kilometers Driven', '').replace(' KMs', '').replace(',', '')
        car_age = request.form.get('Car Age', '')
        fuel_type = request.form.get('Fuel Type', '')
        make = request.form.get('Make', '')
        model_name = request.form.get('Model', '')
        variant = request.form.get('Variant', '')

        if not all([kilometers, car_age, fuel_type, make, model_name, variant]):
            return render_template('index.html', error="❌ Missing input data. Please provide all values.")

        try:
            kilometers = float(kilometers)
            car_age = int(car_age)
        except ValueError:
            return render_template('index.html', error="❌ Invalid input for Kilometers Driven or Car Age. Please enter numbers.")

        # 2. Create input DataFrame
        input_df = pd.DataFrame([{
            'Kilometers Driven': kilometers,
            'Car Age': car_age,
            'Fuel Type': fuel_type,
            'Make': make,
            'Model': model_name,
            'Variant': variant
        }])

        # 3. One-Hot Encode using the loaded encoder
        categorical_cols = ['Fuel Type', 'Make', 'Model', 'Variant']
        for col in categorical_cols:
            input_df[col] = input_df[col].astype(str) #Convert to string to avoid unseen value errors
        input_df_encoded = encoder.transform(input_df[categorical_cols]).toarray()
        encoded_column_names = encoder.get_feature_names_out(categorical_cols)
        input_df_encoded = pd.DataFrame(input_df_encoded, columns=encoded_column_names)
        input_df = input_df.drop(columns=categorical_cols, axis=1)
        input_df = pd.concat([input_df, input_df_encoded], axis=1)

        # 4. Create a DataFrame with ALL the necessary columns (initialized with 0s)
        all_columns = model.feature_names_in_
        final_input_df = pd.DataFrame(0, index=[0], columns=all_columns)

        # 5. Update the DataFrame with the user's input (including encoded values)
        for col in input_df.columns:
            if col in final_input_df.columns:
                final_input_df[col] = input_df[col]

        # 6. Convert to NumPy *after* ALL DataFrame operations are complete
        final_input = final_input_df.astype('float64').to_numpy()

        print("✅ Final Input Shape:", final_input.shape)
        print("✅ Final Input Columns:", final_input_df.columns.tolist())
        print("✅ Final Input dtypes:\n", final_input_df.dtypes)
        print("✅ Final Input Data:\n", final_input_df)
        print("✅ Any NaNs in final_input:", np.isnan(final_input).any())
        print("✅ Type of final_input:", type(final_input))

        predicted_price = model.predict(final_input)[0]

        return render_template('index.html', predicted_price=round(predicted_price, 2))

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return render_template('index.html', error=f"❌ An error occurred during prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True)