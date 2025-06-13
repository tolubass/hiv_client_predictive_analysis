from flask import Flask, render_template, request, jsonify
import joblib  # Using joblib instead of pickle
import numpy as np
import pandas as pd
import os
import logging

# Logging setup
logging.basicConfig(filename='hiv_app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Paths
model_path = r'C:\Users\hp\Desktop\good model\hiv_model_joblib.pkl'  # Updated path to joblib model
template_folder_path = r'C:\Users\hp\Desktop\hiv_model\templates'   # Make sure this folder exists with hiv.html
excel_file_path = r'C:\Users\hp\Desktop\hiv_model\hiv_predictions.xlsx'

# Flask app with correct template folder path
app = Flask(__name__, template_folder=template_folder_path)

# Load model using joblib
model = joblib.load(model_path)

@app.route('/')
def home():
    logging.info("Home page accessed.")
    return render_template("hiv.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from form
        Marital_Status = int(request.form['Marital_Status'])
        Education_Level = int(request.form['Education_Level'])
        Perception_Category = float(request.form['Perception_Category'])
        Years_on_Treatment = float(request.form['Years_on_Treatment'])
        Overall_Comfort_Level = float(request.form['Overall_Comfort_Level'])
        Monthly_Income = float(request.form['Monthly_Income'])
        Household_Size = int(request.form['Household_Size'])
        Treatment_Regimen = int(request.form['Treatment_Regimen'])
        Care_Timeliness = int(request.form['Care_Timeliness'])
        Comfort_StaffInteraction = int(request.form['Comfort_StaffInteraction'])

        input_data = [
            Marital_Status, Education_Level, Perception_Category,
            Years_on_Treatment, Overall_Comfort_Level, Monthly_Income,
            Household_Size, Treatment_Regimen, Care_Timeliness, Comfort_StaffInteraction
        ]
        input_array = np.array(input_data).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_array)[0]
        result = f"Predicted Experience Score: {prediction}"
        logging.info(f"Prediction: {result}")

        # Save to Excel
        df1 = pd.DataFrame([{
            'Marital_Status': Marital_Status,
            'Education_Level': Education_Level,
            'Perception_Category': Perception_Category,
            'Years_on_Treatment': Years_on_Treatment,
            'Overall_Comfort_Level': Overall_Comfort_Level,
            'Monthly_Income': Monthly_Income,
            'Household_Size': Household_Size,
            'Treatment_Regimen': Treatment_Regimen,
            'Care_Timeliness': Care_Timeliness,
            'Comfort_StaffInteraction': Comfort_StaffInteraction,
            'Prediction_Result': prediction
        }])

        if os.path.exists(excel_file_path):
            existing_df = pd.read_excel(excel_file_path)
            df1 = pd.concat([existing_df, df1], ignore_index=True)

        df1.to_excel(excel_file_path, index=False)
        logging.info("Prediction data saved to Excel.")

        return jsonify({'result': result})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)