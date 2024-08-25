from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

# Load the trained model and scaler
model = load_model('my_model.h5')  # Ensure your model is saved as .h5
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

# Define a function to fetch and preprocess stock data
def fetch_and_preprocess_stock_data(symbol, api_key):
    ts = TimeSeries(api_key, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol, interval='1min', outputsize='full')
    
    # Rename columns and preprocess
    columns = ['open', 'high', 'low', 'close', 'volume']
    data.columns = columns
    data['Date'] = data.index.date
    data['Time'] = data.index.time
    market = data.between_time('00:00', '23:59').copy()
    market.sort_index(inplace=True)
    
    # Get the 'close' column and preprocess
    df1 = market.reset_index()['close']
    df1 = scaler.transform(df1.values.reshape(-1, 1))
    
    # Prepare data for prediction
    required_length = 100
    while len(df1) < required_length:
        df1 = np.append(df1, [[0]])
    
    # Reshape for model input
    df1 = df1.reshape(1, df1.shape[0], 1)
    
    return df1, market  # Return the preprocessed market data

# Predict stock prices
def predict_stock_prices(model, symbol, api_key):
    features, data = fetch_and_preprocess_stock_data(symbol, api_key)  # Fetch and preprocess data
    
    # Make predictions
    predictions = model.predict(features)
    predicted_prices = scaler.inverse_transform(predictions)
    
    # Predict the next day's price
    next_day_feature = features[:, -1:, :]  # Use the last available data point
    next_day_prediction = model.predict(next_day_feature)
    next_day_price = scaler.inverse_transform(next_day_prediction)
    
    return predicted_prices, next_day_price, data  # Return both predicted prices and the original data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    api_key = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key
    
    predicted_prices, next_day_price, data = predict_stock_prices(model, symbol, api_key)
    
    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['close'], label='Actual Prices')
    
    # Ensure the predicted prices are aligned correctly
    predicted_dates = [data.index[-1] + timedelta(minutes=i) for i in range(1, len(predicted_prices) + 1)]
    plt.plot(predicted_dates, predicted_prices, label='Predicted Prices', linestyle='--')
    
    # Plot the next day's price
    next_day_date = predicted_dates[-1] + timedelta(days=1)
    plt.plot([next_day_date], next_day_price, label='Next Day Prediction', marker='o', color='red')
    
    plt.title(f'Stock Prices for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Prepare predicted price for rendering
    predicted_price_value = next_day_price.flatten()[0]

    return render_template('result.html', plot_url=plot_url, predicted_price_value=predicted_price_value)

if __name__ == "__main__":
    app.run(debug=True)

