from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from collections import deque

# Load your trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize the Flask application
app = Flask(__name__)

# Initialize a deque to store recent sentiment predictions
sentiment_trends = deque(maxlen=10)

# Define the home route
@app.route('/')
def home():
    return render_template('app.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['text']
    processed_data = vectorizer.transform([data])
    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        result = "The sentiment of your tweet is Negative"
        sentiment_trends.append("Negative")
    elif prediction == 0:
        result = "PThe sentiment of your tweet is positive"
        sentiment_trends.append("Positive")
    else:
        result = "Neutral"
        sentiment_trends.append("Neutral")

    return jsonify({'response': result})

# Define a route for the trends
@app.route('/trends')
def trends():
    return jsonify(list(sentiment_trends))

if __name__ == "__main__":
    app.run(debug=True)
