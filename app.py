from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('car_price_predictor.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    kilometers = float(request.form['Kilometers Driven'].replace(' KMs', '').replace(',', ''))
    car_age = int(request.form['Car Age'])
    fuel_type = request.form['Fuel Type']
    make = request.form['Make']
    model_name = request.form['Model']
    variant = request.form['Variant']

    # Create a DataFrame for the input data
    input_data = {
        'Kilometers Driven': kilometers,
        'Car Age': car_age,
        'Fuel Type': fuel_type,  # Fuel type as a single column for one-hot encoding
        'Make': make,            # Make as a single column for one-hot encoding
        'Model': model_name,
        'Variant': variant
    }

    input_df = pd.DataFrame([input_data])

    # Predict the price using the trained model
    predicted_price = model.predict(input_df)[0]

    # Render the result in the template
    return render_template('index.html', predicted_price=round(predicted_price, 2))

if __name__ == '__main__':
    app.run(debug=True)
