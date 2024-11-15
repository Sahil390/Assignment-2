from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('rf_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pm2_5 = float(request.form['pm2_5'])
    pm10 = float(request.form['pm10'])
    no2 = float(request.form['no2'])
    so2 = float(request.form['so2'])
    o3 = float(request.form['o3'])
    co = float(request.form['co'])
    benzene = float(request.form['benzene'])
    nh3 = float(request.form['nh3'])
    no = float(request.form['no'])
    nox = float(request.form['nox'])
    toluene = float(request.form['toluene'])
    xylene = float(request.form['xylene'])
    dayofweek = int(request.form['dayofweek'])
    month = int(request.form['month'])

    data = pd.DataFrame({
        'PM2.5': [pm2_5],
        'PM10': [pm10],
        'NO2': [no2],
        'SO2': [so2],
        'O3': [o3],
        'CO': [co],
        'Benzene': [benzene],
        'NH3': [nh3],
        'NO': [no],
        'NOx': [nox],
        'Toluene': [toluene],
        'Xylene': [xylene],
        'DayOfWeek': [dayofweek],
        'Month': [month]
    })

    prediction = model.predict(data)
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)