from flask import Flask, request, jsonify
import joblib
import pandas as pd  # Import pandas to handle DataFrame for feature names

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Define the full list of feature columns (retrieved using model.feature_names_in_)
feature_columns = [
    'Brand Name', 'Body Dimension', 'Body SIM', 'Body Weight', 'Comms Bluetooth',
    'Comms GPS', 'Comms Infrared port', 'Comms NFC', 'Comms Radio', 'Comms USB',
    'Comms WLAN', 'Display Protection', 'Display Resolution', 'Display Size',
    'Display Type', 'Features Sensors', 'Launch Announced', 'Launch Status',
    'Camera', 'Memory Call records', 'Memory Card Slot', 'Ram and Storage',
    'Memory Phonebook', 'Misc Models', 'Misc SAR', 'Misc SAR EU',
    'Network Technology', 'Platform CPU', 'Platform Chipset', 'Platform GPU',
    'Platform OS', 'Selfie camera Video', 'Sound 3.5mm jack',
    'Sound Loudspeaker', 'Battery', 'Selfie Camera', 'Main Camera',
    'Accelerometer Sensor', 'Fingerprint Sensor', 'Gyro Sensor',
    'Proximity Sensor', 'Compass Sensor', 'Storage', 'RAM'
]


@app.route('/')
def home():
    return "ML Prediction API is Running!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Log incoming requests
    print(f"Request method: {request.method}")
    print(f"Request data: {request.json if request.method == 'POST' else 'N/A'}")

    if request.method == 'GET':
        return jsonify({
            "status": "success",
            "message": "GET request received! Endpoint is working for debugging."
        })
    elif request.method == 'POST':
        try:
            # Get the input JSON data
            data = request.json

            # Ensure all feature columns are included; set missing ones to default value (0)
            input_features = [float(data.get(col, 0)) for col in feature_columns]

            # Convert input to a pandas DataFrame, including feature column names
            input_df = pd.DataFrame([input_features], columns=feature_columns)

            # Make prediction using the DataFrame
            prediction = model.predict(input_df)

            # Return the prediction result as JSON
            return jsonify({
                'prediction': float(prediction[0]),
                'status': 'success'
            })
        except Exception as e:
            # Catch and return any errors
            return jsonify({
                'error': str(e),
                'status': 'failed'
            })


if __name__ == '__main__':
    app.run(debug=True)
