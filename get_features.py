import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Get the feature names used during training
print(model.feature_names_in_)
