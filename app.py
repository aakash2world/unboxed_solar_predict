# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:27:10 2023

@author: 91939
"""

import numpy as np
from flask import Flask, request, render_template
import pickle
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the saved model
model_filename = 'C:\Ak Data\Study-AK\MLPython\labpractice\Solar\webappsolar\solar_energy_model.pkl'
model = joblib.load(model_filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    cloud_cover = float(request.form['cloud_cover'])
    panel_capacity = float(request.form['panel_capacity'])
    irradiation = float(request.form['irradiation'])
    
    # Predict energy generation
    predicted_energy = model.predict([[temperature, humidity, cloud_cover, panel_capacity, irradiation]])
    
    return render_template('index.html', prediction=f'Predicted energy generation: {predicted_energy[0]:.2f} kWh')

if __name__ == '__main__':
    app.run(debug=True)
