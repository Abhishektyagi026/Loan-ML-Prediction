import re
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)
model = pickle.load('model.pkl', 'rb')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/sub", methods = ['POSt'])
def submit():
    if request.method == "POST":
        Loan_Amount = request.form["Loan Amount"]
        Interest_Rate =request.form["Interest Rate"]
        Annual_Income = request.form["Annual Income"]
        Debt_To_Income_Ratio = request.form["Debt To Income Ratio"]
        FICO_Range = request.form["FICO_Range(Low)"]




if __name__ == '__main__':
    app.run()