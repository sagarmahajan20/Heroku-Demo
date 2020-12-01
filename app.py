import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import request
import flask

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    url = request.args.get('url') 
    return {'object': "object found at 5 meters", "url" : "CAr"}

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return 'Car Detected at 5 metre'


if __name__ == "__main__":
    app.run(debug=True)
