from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# Load models and metadata
stack_model = joblib.load('model/xgb_model.pkl')
kmeans = joblib.load('model/kmeans_model.pkl')
scaler = joblib.load('model/cluster_scaler.pkl')
cluster_features = joblib.load('model/cluster_features.pkl')
X_columns = joblib.load('model/xgb_features.pkl')

# Cluster label mapping
cluster_map = {
    0: 'Low Risk Employees',
    1: 'Moderate Risk Employees',
    2: 'High Risk Employees',
    3: 'Very High Risk Employees'
}

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.static_folder, 'images'), filename)

@app.route('/')
def home():
    return jsonify({'message': 'API is running'})

#  Handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # Feature engineering
    df['ServicePerAge'] = df['LengthService'] / df['Age']
    df['ServiceSquared'] = df['LengthService'] ** 2
    df['AgeSquared'] = df['Age'] ** 2
    df['Age_x_Service'] = df['Age'] * df['LengthService']
    df['AgeDivService'] = df['Age'] / (df['LengthService'] + 1)
    df['IsShortService'] = (df['LengthService'] < 2).astype(int)

    # One-hot encode and align with training columns
    df_encoded = pd.get_dummies(df)
    for col in X_columns:
        if col not in df_encoded:
            df_encoded[col] = 0
    df_encoded = df_encoded[X_columns]

    # Predict absenteeism hours
    predicted_hours = stack_model.predict(df_encoded)[0]

    # Clustering
    df['AbsentHours'] = predicted_hours
    cluster_input = df[cluster_features]
    cluster_scaled = scaler.transform(cluster_input)
    cluster_id = int(kmeans.predict(cluster_scaled)[0])
    cluster_name = cluster_map.get(cluster_id, 'Unknown')

    return jsonify({
        'predicted_hours': float(predicted_hours),
        'cluster_id': cluster_id,
        'cluster_name': cluster_name
    })

if __name__ == '__main__':
    app.run(debug=True)