from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

np.random.seed(1)
W1 = np.random.randn(3, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Page web
@app.route('/')
def home():
    return render_template("index.html")

# Prediction via formulaire
@app.route('/predict_form', methods=['POST'])
def predict_form():
    age = float(request.form['age'])
    salary = float(request.form['salary'])
    visits = float(request.form['visits'])

    X = np.array([[age, salary, visits]])
    X = X / np.array([40, 5000, 10])

    A1 = sigmoid(np.dot(X, W1) + b1)
    A2 = sigmoid(np.dot(A1, W2) + b2)

    result = round(float(A2[0][0]) * 100, 2)

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)