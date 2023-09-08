from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
#from . import db

import json
import numpy as np

import pickle

model = pickle.load(open('models/decisiontree.pkl', 'rb'))

views = Blueprint('views', __name__)

@views.route('/')
def home():

    return render_template('home.html', user= current_user)

@views.route('/home.html', methods=['GET', 'POST'])
def predict():
    gender_hidden = request.form.get('Gender')
    if gender_hidden == 'male':
        gender_value = 1  
    elif gender_hidden =='female': 
        gender_value =2
    else:
        gender_value = 0  

    
    int_features = [float(x) for name, x in request.form.items() if name != 'Gender'] #Convert string inputs to float.
    print("int_features")
    print("Gender Value:", gender_value)
    print("Other Features:", int_features)

    features = [gender_value] + int_features  # Combine gender_value with other features
    features_array = np.array(features).reshape(1, -1)  #Convert to the form [[a, b]] for input to the model
    print("Features Array Shape:", features_array.shape)

    prediction = model.predict(features_array)  # features Must be in the form [[a, b]]
    print("Prediction:", prediction)

    output = round(prediction[0], 2)
    Level= 'Low' if output== 0 else ('High'if output== 1 else 'Medium' )
    return render_template('home.html', prediction_text='Prediction of lung cancer is {}'.format(Level), user=current_user)



