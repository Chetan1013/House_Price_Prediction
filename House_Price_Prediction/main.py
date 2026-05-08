import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np  
app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe=pickle.load(open('RidgeModel.pkl','rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    sqft = request.form.get('sqft')
    bath = request.form.get('bath')
    bhk = request.form.get('bhk')
    print(locations, sqft, bath, bhk)
    input = pd.DataFrame(
    [[locations, float(sqft), int(bath), int(bhk)]],
    columns=['location', 'total_sqft', 'bath', 'bhk']
)
    prediction = pipe.predict(input)[0] * 1e5
    

    return f"₹ {prediction:,.2f}"
def home():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True,port=5001)
