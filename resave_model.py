import joblib

# Load the original model file (make sure it's in the same folder or use full path)
model = joblib.load('logistic_model_protocol4.pkl')

# Re-save with protocol=4 for Python 3.6 compatibility
joblib.dump(model, 'logistic_model_protocol4.pkl', protocol=4)

print("Model re-saved with protocol=4 successfully!")
