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

# Auto-generate EDA plots

def generate_eda_plots():
    img_dir = os.path.join('static', 'images')
    os.makedirs(img_dir, exist_ok=True)
    age_path = os.path.join(img_dir, 'age_distribution.png')
    dept_path = os.path.join(img_dir, 'absenthours_by_dept.png')
    corr_path = os.path.join(img_dir, 'correlation_heatmap.png')
    cluster_path = os.path.join(img_dir, 'cluster_distribution.png')

    # Only regenerate if images are missing to avoid unnecessary work
    if all(os.path.exists(p) for p in [age_path, dept_path, corr_path, cluster_path]):
        return

    df = pd.read_csv('MFGEmployees.csv')

    # --- feature engineering (match predict route) ---
    df['ServicePerAge'] = df['LengthService'] / df['Age']
    df['ServiceSquared'] = df['LengthService'] ** 2
    df['AgeSquared'] = df['Age'] ** 2
    df['Age_x_Service'] = df['Age'] * df['LengthService']
    df['AgeDivService'] = df['Age'] / (df['LengthService'] + 1)
    df['IsShortService'] = (df['LengthService'] < 2).astype(int)

    # Age Distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(df['Age'], bins=30, kde=True, color='#3498db')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(age_path)
    plt.close()

    # AbsentHours by Department
    plt.figure(figsize=(10, 6))
    dept_hours = df.groupby('DepartmentName')['AbsentHours'].sum().sort_values(ascending=False).head(15)
    sns.barplot(x=dept_hours.values, y=dept_hours.index, palette='viridis')
    plt.title('AbsentHours by Department (Top 15)')
    plt.xlabel('Total Absent Hours')
    plt.ylabel('Department')
    plt.tight_layout()
    plt.savefig(dept_path)
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ['Age', 'LengthService', 'AbsentHours']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(corr_path)
    plt.close()

    # Cluster Distribution
    try:
        kmeans_local = joblib.load('model/kmeans_model.pkl')
        scaler_local = joblib.load('model/cluster_scaler.pkl')
        cluster_feats = joblib.load('model/cluster_features.pkl')
        cluster_map_local = {0: 'Low Risk', 1: 'Moderate', 2: 'High', 3: 'Very High'}
        cluster_input = df[cluster_feats]
        cluster_scaled = scaler_local.transform(cluster_input)
        clusters = kmeans_local.predict(cluster_scaled)
        cluster_counts = pd.Series(clusters).map(cluster_map_local).value_counts()

        plt.figure(figsize=(10, 8))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='pastel')
        plt.title('Employee Count per Cluster')
        plt.xlabel('Cluster Group')
        plt.ylabel('Number of Employees')
        plt.tight_layout()
        plt.savefig(cluster_path)
        plt.close()

        # scatter plot Age vs AbsentHours colored by cluster
        scatter_path = os.path.join(img_dir, 'cluster_scatter.png')
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df['Age'], y=df['AbsentHours'], hue=pd.Series(clusters).map(cluster_map_local), palette='Set2', s=60)
        plt.title('Age vs AbsentHours by Cluster')
        plt.xlabel('Age')
        plt.ylabel('Absent Hours')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
    except Exception as e:
        print('Pretrained clustering failed, falling back to new KMeans:', e)
        try:
            scaler_fallback = StandardScaler()
            scaled = scaler_fallback.fit_transform(df[['AbsentHours','Age','LengthService','ServicePerAge','Age_x_Service']].fillna(0))
            kmeans_fb = KMeans(n_clusters=4, random_state=42)
            clusters_fb = kmeans_fb.fit_predict(scaled)
            cluster_map_fb = {0:'Cluster 0',1:'Cluster 1',2:'Cluster 2',3:'Cluster 3'}
            cluster_counts = pd.Series(clusters_fb).map(cluster_map_fb).value_counts()
            # bar plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='pastel')
            plt.title('Employee Count per Cluster')
            plt.xlabel('Cluster Group')
            plt.ylabel('Number of Employees')
            plt.tight_layout()
            plt.savefig(cluster_path)
            plt.close()
            # scatter
            scatter_path = os.path.join(img_dir, 'cluster_scatter.png')
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x=df['Age'], y=df['AbsentHours'], hue=pd.Series(clusters_fb).map(cluster_map_fb), palette='Set2', s=60)
            plt.title('Age vs AbsentHours by Cluster')
            plt.xlabel('Age')
            plt.ylabel('Absent Hours')
            plt.legend(title='Cluster')
            plt.tight_layout()
            plt.savefig(scatter_path)
            plt.close()
            print('Fallback clustering plots generated.')
        except Exception as e2:
            print('Fallback clustering also failed:', e2)

# Generate plots at startup
generate_eda_plots()

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