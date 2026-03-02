from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import random
import os

app = Flask(__name__)

# Load the model - Ensure fire_model.pkl is in the same directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fire_model.pkl')
model = joblib.load(MODEL_PATH)

def prepare_sensor_data(data):
    """Packages raw input into a single structure for the AI."""
    try:
        # Order must match the training: [temp, smoke, load, fuel, vibe]
        structured_data = [
            float(data.get('temp', 40)),
            float(data.get('smoke', 0)),
            float(data.get('load', 300)),
            float(data.get('fuel', 25)),
            float(data.get('vibe', 0.4))
        ]
        return np.array(structured_data).reshape(1, -1)
    except Exception as e:
        print(f"DTO Error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles manual slider input."""
    data = request.get_json()
    formatted_input = prepare_sensor_data(data)
    
    if formatted_input is not None:
        prediction = int(model.predict(formatted_input)[0])
        return jsonify({'risk_level': prediction})
    return jsonify({'error': 'Invalid Data'}), 400

@app.route('/api/live-telemetry', methods=['GET'])
def get_live_telemetry():
    """Triggers a new data snapshot from 'onboard sensors'."""
    # Simulate realistic ship data
    live_snapshot = {
        'temp': round(random.uniform(40, 98), 1),
        'smoke': round(random.uniform(0, 4.8), 2),
        'load': round(random.uniform(200, 580), 0),
        'fuel': round(random.uniform(10, 50), 1),
        'vibe': round(random.uniform(0.1, 1.0), 2)
    }
    
    # Process through AI immediately
    formatted_input = prepare_sensor_data(live_snapshot)
    prediction = int(model.predict(formatted_input)[0])
    
    return jsonify({
        "telemetry": live_snapshot,
        "risk_level": prediction
    })

if __name__ == '__main__':
    app.run(debug=True)