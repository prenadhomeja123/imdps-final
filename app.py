import pickle
import numpy as np
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
from flask import Flask, request, jsonify, render_template , redirect , url_for
with open('heart.pkl' , 'rb') as file:
    heart_model=pickle.load(file)
with open('kidneyStone.pkl' , 'rb') as file:
    kidneyStone_model=pickle.load(file)
with open('parkinsons.pkl' , 'rb') as file:
    parkinsons_model=pickle.load(file)
with open('hepatitisC.pkl' , 'rb') as file:
    hepatitisC_model=pickle.load(file)
with open('liver.pkl' , 'rb') as file:
    liver_model=pickle.load(file)
with open('lung_cancer_svm.pkl' , 'rb') as file:
    lung_cancer_model=pickle.load(file)
with open('strokeee.pkl', 'rb') as file:
    stroke_model=pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/homee')
def homee():
    return render_template('home.html')
@app.route('/heart')
def heart():
    return render_template('heart.html')
@app.route('/diabetes')
def diabetes():
    return render_template('index.html')
@app.route('/liver')
def liver():
    return render_template('liver.html')


@app.route('/lung')
def lung():
    return render_template('lung.html')
@app.route('/kidney')
def kidney():
    return render_template('kidney.html')
@app.route('/parkinsons')
def parkinsons():
    return render_template('parkinsons.html')
@app.route('/hepatitisC')
def hepatitisC():
    return render_template('hepatitisC.html')
@app.route('/strok')
def stroke():
    return render_template('strok.html')
@app.route('/predict/heart', methods=['POST'])
def predictheart():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = heart_model.predict(final_features)
    output = prediction[0]
    if output == 1:
        return render_template('heart.html', prediction_text='The person is likely to have heart disease.')
    else:
        return render_template('heart.html', prediction_text='The person is not likely to have heart disease.')


@app.route('/predict/diabetes', methods=['POST'])
def predictdiabetes():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        # Assuming diabetes_model is loaded and ready for prediction
        diabetes_prediction = model.predict(final_features)
        diabetes_output = diabetes_prediction[0]

        if diabetes_output == 1:
            return render_template('index.html', prediction_text='The person is likely to have diabetes.')
        else:
            return render_template('index.html', prediction_text='The person is not likely to have diabetes.')

@app.route('/predict/parkinsons', methods=['POST'])
def predictparkinsons():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        # Assuming diabetes_model is loaded and ready for prediction
        parkinsons_prediction = parkinsons_model.predict(final_features)
        parkinsons_output = parkinsons_prediction[0]

        if parkinsons_output == 1:
            return render_template('parkinsons.html', prediction_text='The person is likely to have diabetes.')
        else:
            return render_template('parkinsons.html', prediction_text='The person is not likely to have diabetes.')
@app.route('/predict/kidney', methods=['POST'])
def predictkidneystone():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = kidneyStone_model.predict(final_features)
    output = prediction[0]
    if output == 1:
        return render_template('kidney.html', prediction_text='The person is likely to have kidney stone.')
    else:
        return render_template('kidney.html', prediction_text='The person is not likely to have kidney stone.')

@app.route('/predict/hepatitisC', methods=['POST'])
def predicthepatitisC():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = hepatitisC_model.predict(final_features)
    output = prediction[0]
    if output == 1:
        return render_template('hepatitisC.html', prediction_text='The person is likely to have kidney stone.')
    else:
        return render_template('hepatitisC.html', prediction_text='The person is not likely to have kidney stone.')
@app.route('/predict/liver', methods=['POST'])
def predictliver():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = liver_model.predict(final_features)
    output = prediction[0]
    if output == 1:
        return render_template('liver.html', prediction_text='The person is likely to have liver disease.')
    else:
        return render_template('liver.html', prediction_text='The person is not likely to have liver disease.')

@app.route('/predict/stroke', methods=['POST'])
def predictstroke():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = stroke_model.predict(final_features)
    output = prediction[0]
    if output == 1:
        return render_template('strok.html', prediction_text='The person is likely to have stroke .')
    else:
        return render_template('strok.html', prediction_text='The person is not likely to have stroke .')

@app.route('/predict/lung', methods=['POST'])
def predictlung_cancer():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = lung_cancer_model.predict(final_features)
    output = prediction[0]
    if output == 1:
        return render_template('lung.html', prediction_text='The person is likely to have lung cancer .')
    else:
        return render_template('lung.html', prediction_text='The person is not likely to have lung cancer .')








if __name__ == '__main__':
    app.run(debug=True)
