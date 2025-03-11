import customtkinter as ctk
import joblib
import numpy as np

# Load trained models
xgb_model = joblib.load('saved_models/best_xgb_model_selected.pkl')
cat_model = joblib.load('saved_models/best_catboost_model_selected.pkl')

# Define selected features with full descriptions
features = {
    'age': "Age (years)",
    'sex': "Sex (Male=1, Female=0)",
    'cp': "Chest Pain Type (0-3)(cp)",
    'trestbps': "Resting Blood Pressure (mm Hg)(trestbps)",
    'chol': "Cholesterol (mg/dL)(chol)",
    # 'thalch':"Maximum Heart rate Achieved(thalch)",
    'exang': "Exercise-Induced Angina (Yes=1, No=0)(exang)",
    'oldpeak': "ST Depression Induced by Exercise(oldpeak)",
    'slope': "Slope of Peak Exercise ST Segment (0-2)(slope)",
    'ca': "Number of Major Vessels Colored (0-3)(ca)",
    'thal': "Thalassemia Type (0-3)(thal)"
}

features={
    'cp': "Chest Pain Type (0-3)(cp)",
    'thal': "Thalassemia Type (0-3)(thal)",
    'ca': "Number of Major Vessels Colored (0-3)(ca)",
'slope': "Slope of Peak Exercise ST Segment (0-2)(slope)",
'exang': "Exercise-Induced Angina (Yes=1, No=0)(exang)",
    'oldpeak': "ST Depression Induced by Exercise(oldpeak)",
'sex': "Sex (Male=1, Female=0)",
'thalch':"Maximum Heart rate Achieved(thalch)",
'age': "Age (years)",
'chol': "Cholesterol (mg/dL)(chol)"
}

# Create main window with light mode
ctk.set_appearance_mode("light")
root = ctk.CTk()
root.title("Heart Disease Prediction")
root.geometry("600x600")

# Dictionary to store input fields
entries = {}

# Create input fields
for i, (feature, label_text) in enumerate(features.items()):
    label = ctk.CTkLabel(root, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e")

    if feature in ['sex', 'cp', 'exang', 'slope', 'ca', 'thal']:
        # Dropdown for categorical values
        options = {
            'sex': ["Male", "Female"],
            'cp': ["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"],
            'exang': ["1:Yes", "0:No"],
            'slope': ["0: Upsloping", "1: Flat", "2: Downsloping"],
            'ca': ["0", "1", "2", "3"],
            'thal': ["0: Normal", "1: Fixed Defect", "2: Reversible Defect", "3: Unknown"]
        }

        dropdown = ctk.CTkComboBox(root, values=options[feature])
        dropdown.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        entries[feature] = dropdown
    else:
        # Numeric entry for continuous variables
        entry = ctk.CTkEntry(root)
        entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        entries[feature] = entry


# Prediction function
def predict():
    try:
        user_input = []
        for feature in features.keys():
            value = entries[feature].get()

            # Convert categorical dropdowns to numeric
            if feature == 'sex':
                value = 1 if value == "Male" else 0
            elif feature in ['exang', 'ca', 'thal', 'slope', 'cp']:
                value = int(value.split(':')[0])  # Extract numeric value
            else:
                value = float(value)

            user_input.append(value)

        input_array = np.array(user_input).reshape(1, -1)

        # Predictions
        xgb_pred = xgb_model.predict(input_array)[0]
        cat_pred = cat_model.predict(input_array)[0]

        # Display result
        xgb_text = "Heart Disease Predicted" if xgb_pred == 1 else "No Heart Disease Predicted"
        cat_text = "Heart Disease Predicted" if cat_pred == 1 else "No Heart Disease Predicted"

        xgb_color = "red" if xgb_pred == 1 else "green"
        cat_color = "red" if cat_pred == 1 else "green"

        result_label.configure(
            text=f"XGBoost: {xgb_text}\nCatBoost: {cat_text}",
            text_color=xgb_color if xgb_pred == 1 else cat_color  # If any model predicts heart disease, use red
        )

    except Exception as e:
        result_label.configure(text=f"Error: {str(e)}")


# Predict button
predict_button = ctk.CTkButton(root, text="Predict", command=predict)
predict_button.grid(row=len(features), column=0, columnspan=2, pady=20)
result_pane=ctk.CTkFrame(root,fg_color="#fafafa",height=50,width=300,corner_radius=10)
result_pane.grid(row=len(features) + 1, column=0, columnspan=3, pady=10)
# Result label
result_label = ctk.CTkLabel(result_pane, text="", font=("Arial", 14))
result_label.place(x=10,y=10)

# Run application
root.mainloop()
