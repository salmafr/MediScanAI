import numpy as np
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)


model = pickle.load(open('models/decisiontree.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predict():
    gender_hidden = request.form.get('Gender')
    gender_value = 1 if gender_hidden == 'male' else 0


    int_features = [float(x) for name, x in request.form.items() if name != 'Gender'] #Convert string inputs to float.
    features = [gender_value] + int_features  # Combine gender_value with other features
    features_array = np.array(features).reshape(1, -1)  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features_array)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)
    Level= 'Low' if output== 0 else ('High'if output== 1 else 'Medium' )
    return render_template('predict.html', prediction_text='Prediction of lung cancer is {}'.format(Level))


if __name__ == "__main__":
    app.run()