from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('flight_delay_model.joblib')

# Load the list of encoded columns
encoded_columns = joblib.load('encoded_columns.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            month = int(request.form['month'])
            day_of_month = int(request.form['day_of_month'])
            day_of_week = int(request.form['day_of_week'])
            unique_carrier = request.form['unique_carrier']
            origin = request.form['origin']
            dest = request.form['dest']
            crs_dep_time = int(request.form['crs_dep_time'])
            distance = float(request.form['distance'])
            crs_elapsed_time = float(request.form['crs_elapsed_time'])
            dep_delay = float(request.form['dep_delay'])
            cancelled = int(request.form['cancelled'])
            diverted = int(request.form['diverted'])

            # Create a DataFrame for the new input, similar to how training data was structured
            input_data = pd.DataFrame([[
                month, day_of_month, day_of_week, unique_carrier, origin, dest,
                crs_dep_time, distance, crs_elapsed_time, dep_delay, cancelled, diverted
            ]], columns=[
                'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER',
                'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DISTANCE',
                'CRS_ELAPSED_TIME', 'DEP_DELAY', 'CANCELLED', 'DIVERTED'
            ])

            # Apply one-hot encoding to the input data
            input_data_encoded = pd.get_dummies(input_data, columns=['UNIQUE_CARRIER', 'ORIGIN', 'DEST'], drop_first=True)

            # Reindex the input_data_encoded to match the columns the model was trained on
            # Fill missing columns (if any) with 0 and align the order
            input_data_aligned = input_data_encoded.reindex(columns=encoded_columns, fill_value=0)

            # Make prediction
            prediction = model.predict(input_data_aligned)[0]
            prediction_proba = model.predict_proba(input_data_aligned)[0]

            result = "Delayed" if prediction == 1 else "On-time"
            delay_probability = prediction_proba[1] * 100 # Probability of delay

            return render_template('index.html', prediction_result=result, delay_proba=f"{delay_probability:.2f}%")

        except Exception as e:
            return render_template('index.html', prediction_result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)