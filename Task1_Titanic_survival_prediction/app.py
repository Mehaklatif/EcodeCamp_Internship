# app.py
from flask import Flask, request, render_template, redirect
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
loaded_model = pickle.load(open('titanic_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    pclass = int(request.form['pclass'])
    sex = request.form['sex'].lower()
    if sex == 'female':
        sex = 0
    elif sex == 'male':
        sex = 1
    else:
        return "Invalid sex input. Please enter 'male' or 'female'."

    age = int(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    embarked = request.form['embarked'].lower()
    if embarked == 'cherbourg':
        embarked = 0
    elif embarked == 'queenstown':
        embarked = 1
    elif embarked == 'southampton':
        embarked = 2
    else:
        return "Invalid embarkation port input. Please enter 'Cherbourg', 'Queenstown', or 'Southampton'."

    # Convert input to a NumPy array for prediction
    user_input = [pclass, sex, age, sibsp, parch, fare, embarked]
    input_array = np.array(user_input).reshape(1, -1)

    # Make the prediction
    prediction = loaded_model.predict(input_array)

    # Interpret the prediction
    if prediction[0] == 0:
        result = "Prediction: Did not survive"
    else:
        result = "Prediction: Survived"

    # Prepare input data for display
    input_data = {
        'pclass': pclass,
        'sex': 'Female' if sex == 0 else 'Male',
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': 'Cherbourg' if embarked == 0 else 'Queenstown' if embarked == 1 else 'Southampton'
    }

    return render_template('result.html', prediction_text=result, input_data=input_data)

if __name__ == "__main__":
    app.run(debug=True)
