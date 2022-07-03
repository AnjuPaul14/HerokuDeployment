import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('./model.pickle','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    if output == 1 :
        o1 = 'Setosa'
    elif output == 2:
        o1 = 'Versicolor'
    else:
        o1 = 'Virginica'
    

    return render_template('index.html', prediction_text='The category of the flower is  {}'.format(o1))

if __name__ == "__main__":
    app.run(port=5000, debug=True)