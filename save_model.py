import joblib
import tarfile
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load your data
data = pd.read_csv('cleaned_combined_dataset1.csv')

# 2. Data Preprocessing and Feature Engineering
# ... (Your code to clean, transform, and engineer features) ...

# 3. Identify Categorical Columns (CRUCIAL)
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()  # All object type columns

# 4. Convert Numerical Columns to Numeric
numerical_cols = data.select_dtypes(include=['number']).columns.tolist()  # All numeric type columns
for col in numerical_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# 5. Handle Missing Values in Numerical Columns
data[numerical_cols] = data[numerical_cols].fillna(0)  # Or use a more sophisticated method

# 6. Ensure Consistent Categorical Column Order and One-Hot Encode
for col in categorical_cols:
    data[col] = data[col].astype(str).astype('category')
    data[col] = data[col].cat.reorder_categories(data[col].unique(), ordered=False)

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(data[categorical_cols])

encoded_data = encoder.transform(data[categorical_cols]).toarray()
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
data = data.drop(columns=categorical_cols)
data = pd.concat([data, encoded_df], axis=1)

# 7. Define features (X) and target (y)
target_column = 'Price'
if target_column not in data.columns:
    raise KeyError(f"Target column '{target_column}' not found in data after preprocessing. Check your data and preprocessing steps.")

y = data[target_column]
X = data.drop(target_column, axis=1, errors='ignore')

# 8. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Initialize and train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ... (Save model and encoder as before)
# 10. Save the trained model
joblib.dump(model, "model/car_price_predictor.pkl")

# 11. Compress the model folder into a .tar.gz file
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model", arcname=".")

print("✅ Model and encoder successfully saved in model.tar.gz")