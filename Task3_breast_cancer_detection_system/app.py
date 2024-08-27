from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    features = [float(request.form.get(key)) for key in [
        'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
        'concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean','radius_se',
        'texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
        'concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst',
        'perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst',
        'concave_points_worst','symmetry_worst','fractal_dimension_worst'    
    ]]
    
    final_features = [np.array(features)]
    prediction = model.predict(final_features)[0]
    prediction_proba = model.predict_proba(final_features)[0]

    output = 'Patient has been diagnosed with malignant cancer.' if prediction == 0 else 'Patient has been diagnosed with Benign cancer.'
    benign_prob = round(prediction_proba[1] * 100, 2)
    malignant_prob = round(prediction_proba[0] * 100, 2)

    return render_template('index.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
