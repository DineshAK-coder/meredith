import os
import joblib
import numpy as np
import random
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Initialize Flask (Point to templates in the root)
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))
CORS(app)

# --- MODEL PATHING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'fire_model.pkl'))

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ ThermalShield AI Kernel Online")
except Exception as e:
    print(f"⚠️ Model Load Warning: {e}")

def get_risk_level(t, s, l, f):
    """Smart Logic: AI Prediction with Manual Safety Overrides"""
    # 1. Base Logic (Safety Net)
    if t > 100 or (t > 85 and s > 3.0):
        level = 2  # CRITICAL
    elif t > 75 or s > 1.5:
        level = 1  # ELEVATED
    else:
        level = 0  # OPTIMAL

    # 2. AI Refinement (If model is loaded)
    if model:
        try:
            features = np.array([[float(t), float(s), float(l), float(f)]])
            ai_p = int(model.predict(features)[0])
            # If AI detects a higher risk than the base logic, trust the AI
            if ai_p > level: level = ai_p
        except:
            pass
    return level

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        t, s = float(data.get('temp', 40)), float(data.get('smoke', 0))
        l, f = float(data.get('load', 300)), float(data.get('fuel', 25))
        
        return jsonify({'risk_level': get_risk_level(t, s, l, f)})
    except Exception as e:
        return jsonify({'error': str(e), 'risk_level': 0}), 400

@app.route('/api/live-telemetry')
def live_telemetry():
    t = round(random.uniform(30.0, 115.0), 1)
    s = round(random.uniform(0.0, 5.0), 2)
    l = round(random.uniform(100.0, 600.0), 0)
    f = round(random.uniform(10.0, 60.0), 1)
    
    return jsonify({
        'telemetry': {'temp': t, 'smoke': s, 'load': l, 'fuel': f},
        'risk_level': get_risk_level(t, s, l, f)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)