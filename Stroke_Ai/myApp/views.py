from django.shortcuts import render
import numpy as np
import joblib
import lime
import lime.lime_tabular
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Feature names used in the model
feature_names = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# Load training data for LIME
X_train = pd.read_csv(r'C:\Users\Hp\OneDrive\Desktop\stroke_ai\myProject\X_train.csv')

# Create global LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=['No Stroke', 'Stroke'],
    mode='classification'
)

def home(request):
    return render(request, 'home.html')

def format_reasons(raw_reasons):
    formatted = []
    for reason in raw_reasons:
        try:
            if ">" in reason:
                feature, rest = reason.split(">")
                threshold, weight = rest.split(":")
                sentence = f"The value of {feature.strip()} is greater than {threshold.strip()} (contribution: {weight.strip()})."
            elif "<=" in reason:
                feature, rest = reason.split("<=")
                threshold, weight = rest.split(":")
                sentence = f"The value of {feature.strip()} is less than or equal to {threshold.strip()} (contribution: {weight.strip()})."
            else:
                feature, weight = reason.split(":")
                sentence = f"The feature '{feature.strip()}' contributed {weight.strip()} to the prediction."
            formatted.append(sentence)
        except Exception as e:
            print(f"Error formatting reason: {reason}, error: {e}")
            continue
    return formatted

def result(request):
    model_path = r'C:\Users\Hp\OneDrive\Desktop\stroke_ai\myProject\stroke_rf.pkl'

    with open(model_path, 'rb') as file:
        model = joblib.load(file)

    
    if request.method == 'POST':
        try:
            # Collect user inputs
            gender = int(request.POST['gender'])
            age = float(request.POST['age'])
            hypertension = int(request.POST['hypertension'])
            heart_disease = int(request.POST['heart_disease'])
            ever_married = int(request.POST['ever_married'])
            work_type = int(request.POST['work_type'])
            residence_type = int(request.POST['Residence_type'])
            avg_glucose_level = float(request.POST['avg_glucose_level'])
            bmi = float(request.POST['bmi'])
            smoking_status = int(request.POST['smoking_status'])

            features = [
                gender, age, hypertension, heart_disease, ever_married,
                work_type, residence_type, avg_glucose_level, bmi, smoking_status
            ]


            input_df = pd.DataFrame([features], columns=feature_names)
            prediction = model.predict(input_df)[0]

            ans = "You are not at risk of having a stroke." if prediction == 0 else "You are at risk of having a stroke."

     
            exp = explainer.explain_instance(
                data_row=input_df.iloc[0],
                predict_fn=model.predict_proba
            )

            # Get top 5 feature contributions
            raw_reasons = [f"{feature}: {round(weight, 5)}" for feature, weight in exp.as_list()[:5]]
            reasons_list = format_reasons(raw_reasons)

            return render(request, 'result.html', {'ans': ans, 'reasons_list': reasons_list})

        except Exception as e:
            print("Error during prediction:", e)
            return render(request, 'home.html', {'error': 'Invalid input. Please try again.'})

    return render(request, 'home.html')
