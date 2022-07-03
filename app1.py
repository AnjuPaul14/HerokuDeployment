from flask import Flask, jsonify, request
import pickle
import pandas as pd

app1 = Flask(__name__)

@app1.route('/', methods = ['GET','POST'])
def home():
    if(request.method == 'GET'):
        data = "Welcome to Flower Type Prediction App"
        return jsonify({'data' : data})

@app1.route('/predict/')
def type_predict():
    model = pickle.load(open('model.pickle','rb'))
    sepal_length = request.args.get('sepal_length')
    sepal_width = request.args.get('sepal_width')
    petal_length = request.args.get('petal_length')
    petal_width = request.args.get('petal_width')

    df = pd.DataFrame({'sepal.length':[sepal_length], 'sepal.width':[sepal_width],'petal.length':[petal_length],'petal.width':[petal_width]})
    pred_type = model.predict(df)
    output = pred_type[0]
    if output == 1 :
        o1 = 'Setosa'
    elif output == 2:
        o1 = 'Versicolor'
    else:
        o1 = 'Virginica'

    return jsonify({'Variety': str(o1)})

if __name__ == '__main__':
    app1.run(debug = True)
    