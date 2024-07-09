from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the pre-trained RandomForest model
model = joblib.load('RandomForest.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        monthly_inhand_salary = float(request.form['Monthly_Inhand_Salary'])
        interest_rate = float(request.form['Interest_Rate'])
        changed_credit_limit = float(request.form['Changed_Credit_Limit'])
        credit_mix = float(request.form['Credit_Mix'])
        outstanding_debt = float(request.form['Outstanding_Debt'])
        credit_history_age = float(request.form['Credit_History_Age'])
        monthly_balance = float(request.form['Monthly_Balance'])
        
        # Create a feature array in the correct order
        features = np.array([[monthly_inhand_salary, interest_rate, changed_credit_limit, 
                              credit_mix, outstanding_debt, credit_history_age, monthly_balance]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Map prediction to category
        if prediction == 0:
            category = "Bad"
        elif prediction == 1:
            category = "Standard"
        else:
            category = "Good"
        
        return render_template('index.html', prediction_text=f'Predicted Category: {category}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
