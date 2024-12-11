import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'C:\\Users\\naran\\Desktop\\Final_clean_cleaned.csv'
df = pd.read_csv(file_path, low_memory=False)


# Function to adjust prices based on the thresholds
def adjust_price(x):
    if x > 5000:
        return x / 100
    elif x > 2000:
        return x / 20
    elif x > 1000:
        return x / 5
    return x


# Process the 'Misc Price' column
def process_misc_price(x):
    try:
        price_matches = re.findall(r'\d+', str(x))
        if price_matches:
            numeric_price = float(price_matches[0])
            if 'INR' in str(x):
                return numeric_price / 80.0
            elif 'EUR' in str(x):
                return numeric_price
            elif 'USD' in str(x):
                return numeric_price * 0.95
            else:
                return adjust_price(numeric_price)
        return 0.0
    except Exception as e:
        print(f"Error processing price: {x}, Error: {e}")
        return 0.0


df['Misc Price'] = df['Misc Price'].apply(process_misc_price).astype(float)

# Handle outliers in 'Misc Price'
lower_limit = df['Misc Price'].quantile(0.01)
upper_limit = df['Misc Price'].quantile(0.99)
df = df[(df['Misc Price'] >= lower_limit) & (df["Misc Price"] <= upper_limit)]

# Handle missing values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

# Convert boolean-like columns to integers (assumes values are '1/0' or boolean-like strings)
bool_columns = [
    'Comms Bluetooth', 'Comms GPS', 'Comms Infrared port', 'Comms NFC',
    'Comms Radio', 'Comms WLAN', 'Sound 3.5mm jack', 'Sound Loudspeaker',
    'Accelerometer Sensor', 'Fingerprint Sensor', 'Gyro Sensor',
    'Proximity Sensor', 'Compass Sensor'
]

for col in bool_columns:
    if col in df.columns:  # Check for the column's existence
        df[col] = df[col].fillna(0).astype(int)  # Fill missing values before conversion

# Encode remaining categorical variables using LabelEncoder
label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].fillna('Unknown'))

# Define features and target
X = df.drop(columns=['Misc Price', 'Model Name'], errors='ignore')
y = df['Misc Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, random_state=10)  # Reduced n_estimators for speed
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Display predictions (pair actual with predicted)
print("Predicted, Original")
for pred, org in zip(y_pred, y_test.values):
    print(f"{pred}, {org}")

print(f'y_pred max: {y_pred.max()}')
print(f'y_test max: {y_test.max()}')

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-Squared: {r2:.2f}")

import joblib

# Save the trained model
joblib.dump(model, "random_forest_model.pkl")
print("Model saved as random_forest_model.pkl")

