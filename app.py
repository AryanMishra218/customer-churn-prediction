from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained models and preprocessing objects
print("🔄 Loading trained models...")
try:
    # Load the best model (XGBoost)
    model = joblib.load('models/xgboost.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model, scaler, label_encoders = None, None, None

# HOME PAGE ROUTE
@app.route('/')
def home():
    """
    This function runs when user visits the home page
    It displays the main prediction form
    """
    return render_template('index.html')

# PREDICTION API ROUTE
# PREDICTION API ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    """
    This function runs when user clicks "Predict" button
    It receives customer data, makes prediction, and returns result
    """
    try:
        # STEP 1: Get data from the form
        data = request.form
        
        # STEP 2: Extract and convert form data to correct types
        credit_score = int(data['credit_score'])
        geography = data['geography']
        gender = data['gender']
        age = int(data['age'])
        tenure = int(data['tenure'])
        balance = float(data['balance'])
        num_products = int(data['num_products'])
        has_credit_card = int(data['has_credit_card'])
        is_active_member = int(data['is_active_member'])
        estimated_salary = float(data['estimated_salary'])
        
        # STEP 3: Convert text features to numbers
        geography_encoded = label_encoders['Geography'].transform([geography])[0]
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        
        # STEP 4: Create feature array in EXACT same order as training
        features = np.array([[
            credit_score,
            geography_encoded,
            gender_encoded,
            age,
            tenure,
            balance,
            num_products,
            has_credit_card,
            is_active_member,
            estimated_salary
        ]])
        
        # STEP 5: Scale the features
        features_scaled = scaler.transform(features)
        
        # STEP 6: Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # STEP 7: Convert NumPy types to Python native types (FIX FOR JSON ERROR)
        prediction = int(prediction)  # Convert to Python int
        churn_probability = float(round(prediction_proba[1] * 100, 2))  # Convert to Python float
        stay_probability = float(round(prediction_proba[0] * 100, 2))  # Convert to Python float
        
        result = {
            'prediction': prediction,
            'churn_probability': churn_probability,
            'stay_probability': stay_probability,
            'message': 'Customer will LEAVE the bank ⚠️' if prediction == 1 else 'Customer will STAY with the bank ✅',
            'risk_level': get_risk_level(churn_probability)
        }
        
        # STEP 8: Return result as JSON
        return jsonify(result)
    
    except Exception as e:
        # If any error occurs, return error message
        return jsonify({'error': str(e)}), 400

def get_risk_level(churn_prob):
    """
    Helper function to categorize risk level based on churn probability
    """
    if churn_prob < 30:
        return 'Low Risk 🟢'
    elif churn_prob < 60:
        return 'Medium Risk 🟡'
    else:
        return 'High Risk 🔴'

# Run the Flask app
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Customer Churn Prediction API")
    print("="*60)
    print("📍 Server running at: http://127.0.0.1:5000")
    print("💡 Open this URL in your browser to use the app")
    print("⏹️  Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Start the server
    app.run(debug=True, host='0.0.0.0', port=5000)