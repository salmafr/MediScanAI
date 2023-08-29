from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from . import db

import json
import numpy as np

import pickle

model = pickle.load(open('models/decisiontree.pkl', 'rb'))

views = Blueprint('views', __name__)
@views.route('/')
def home():
    return render_template('home.html', user=current_user)

@views.route('/predict', methods=['POST'])
def predict():
    gender_hidden = request.form.get('Gender')
    gender_value = 1 if gender_hidden == 'male' else 0


    int_features = [float(x) for name, x in request.form.items() if name != 'Gender'] #Convert string inputs to float.
    features = [gender_value] + int_features  # Combine gender_value with other features
    features_array = np.array(features).reshape(1, -1)  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features_array)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)
    Level= 'Low' if output== 0 else ('High'if output== 1 else 'Medium' )
    return render_template('home.html', prediction_text='Prediction of lung cancer is {}'.format(Level), user=current_user)


