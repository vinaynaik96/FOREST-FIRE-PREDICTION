# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('forest_fire.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        temp = int(request.form['Temparature'])
        oxy = int(request.form['Oxygen'])
        hum = int(request.form['Humidity'])
        final=np.array([[temp,oxy,hum]])
        prediction=model.predict_proba(final)
        output='{0:.2%}'.format(prediction[0][1], 2)
        return render_template('base.html',pred=output)

if __name__ == '__main__':
	app.run(debug=True, port=5500)