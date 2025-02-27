from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the form input (string) and convert to list of floats
    features_str = request.form['features']  # Get the raw string input from the form
    features = list(map(float, features_str.split(',')))  # Convert the string to a list of floats

    # Send the features to FastAPI for prediction
    response = requests.post("http://127.0.0.1:8000/predict/", json={"features": features})
    
    if response.status_code == 200:
        prediction = response.json().get('prediction', 'Erreur dans la prédiction')
    else:
        prediction = 'Erreur de communication avec FastAPI'
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

