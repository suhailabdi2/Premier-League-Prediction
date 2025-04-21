from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)

# Load the data
df = pd.read_csv('matches.csv')

# Load or train the model
model_path = 'premier_league_model.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    # Prepare features and target
    features = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
    target = 'result'
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[features], df[target])
    joblib.dump(model, model_path)

@app.route('/')
def home():
    # Get unique teams for dropdown
    teams = sorted(df['team'].unique())
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    
    # Get team stats
    home_stats = df[df['team'] == home_team].iloc[-1]
    away_stats = df[df['team'] == away_team].iloc[-1]
    
    # Prepare features for prediction
    features = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
    home_features = home_stats[features].values.reshape(1, -1)
    away_features = away_stats[features].values.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(home_features)[0]
    
    # Map prediction to result
    result_map = {'W': 'Home Win', 'D': 'Draw', 'L': 'Away Win'}
    prediction_text = result_map[prediction]
    
    return jsonify({
        'prediction': prediction_text,
        'home_team': home_team,
        'away_team': away_team
    })

if __name__ == '__main__':
    app.run(debug=True) 