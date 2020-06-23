from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def levelpredictor(list_to_predict):
    to_predict= np.array(list_to_predict).reshape(1,11)
    # loading the classifier
    classifier = pickle.load(open('lgb.pkl', 'rb'))
    result=classifier.predict(to_predict)
    return result

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        list_to_predict = request.form.to_dict()
        list_to_predict = list(list_to_predict.values())
        list_to_predict = list(map(int, list_to_predict))
        result=levelpredictor(list_to_predict)
        if int(result)==1:
            prediction = 'FIRST LEVEL INTEREST RATE(2%-5%), call your bank for more details'
        elif int(result) == 2:
            prediction = 'SECOND LEVEL INTEREST RATE(between(5%-10%)), call your bank for more details'
        elif int(result) == 3:
            prediction = 'THIRD LEVEL INTEREST RATE(between(10%-15%)) , call your bank for more details'
        else:
            prediction = 'You are not entitled to a loan'

        return  render_template("result.html", prediction=prediction)

if __name__=='__main__':
    app.run(debug=True)


