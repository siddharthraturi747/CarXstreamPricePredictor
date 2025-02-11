import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
df = pd.read_csv('cleaned_combined_dataset1.csv')
# Ensure 'Kilometers Driven' is treated as a string before cleaning
df['Kilometers Driven'] = df['Kilometers Driven'].astype(str)

# Remove ' KMs' and ',' from the strings
df['Kilometers Driven'] = df['Kilometers Driven'].str.replace(' KMs', '').str.replace(',', '')

# Convert the cleaned strings to numeric
df['Kilometers Driven'] = pd.to_numeric(df['Kilometers Driven'], errors='coerce')

# Handle NaN values (optional, depending on the dataset)
df['Kilometers Driven'].fillna(0, inplace=True)

df['Car Age'] = 2023 - df['Year']

# Select features and target
X = df[['Kilometers Driven', 'Car Age', 'Fuel Type', 'Make', 'Model', 'Variant']]
y = df['Price']

# One-hot encoding for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('fuel_type', OneHotEncoder(), ['Fuel Type']),
        ('make', OneHotEncoder(), ['Make']),
        ('model', OneHotEncoder(), ['Model']),
        ('variant', OneHotEncoder(), ['Variant']),
        ('scaler', StandardScaler(), ['Kilometers Driven', 'Car Age'])
    ])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model and preprocessing pipeline
joblib.dump(pipeline, 'car_price_predictor.pkl')

print("Model trained and saved as 'car_price_predictor.pkl'")
