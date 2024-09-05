from flask import Flask, request, jsonify, render_template
import pickle

# Load your trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize the Flask application
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['text']
    processed_data = vectorizer.transform([data])
    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        result = "The sentiment of your text is Negative."
    elif prediction == 0:
        result = "The sentiment of your text is Positive."
    else:
        result = "The sentiment of your text is Neutral."

    return render_template('index.html', prediction_text=result)

# Define an API route for JSON input
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    processed_data = vectorizer.transform([data['text']])
    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        result = "The sentiment of your tweet is Negative."
    elif prediction == 0:
        result = "The sentiment of your tweet is Positive."
    else:
        result = "The sentiment of your tweet is Neutral."

    return jsonify({'response': result})

if __name__ == "__main__":
    app.run(debug=True)
