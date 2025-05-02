import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load all models
@st.cache_resource
def load_models():
    models = {
        'lr': joblib.load('Models/best_lr_model.pkl'),
        'nb': joblib.load('Models/best_nb_model.pkl'),
        'rf': joblib.load('Models/best_rf_model.pkl'),
        'svm': joblib.load('Models/best_svm_model.pkl'),
        'ann': load_model('Models/optimized_ann_model.h5')
    }
    return models

models = load_models()

# Create input widgets
st.title('Depression Prediction using Ensemble Model')
st.write('Enter the patient details:')

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=10, max_value=100, value=20)
    academic_pressure = st.slider('Academic Pressure (1-5)', 1, 5, 3)
    study_satisfaction = st.slider('Study Satisfaction (1-5)', 1, 5, 3)
    
with col2:
    dietary_habits = st.selectbox('Dietary Habits', ['Healthy', 'Moderate', 'Unhealthy'])
    suicidal_thoughts = st.selectbox('Have you ever had suicidal thoughts?', ['Yes', 'No'])
    family_history = st.selectbox('Family History of Mental Illness', ['Yes', 'No'])
    financial_stress = st.slider('Financial Stress (1-5)', 1, 5, 3)

# Convert categorical features to numerical values
def encode_features(diet, suicidal, family_hist):
    # Dietary Habits: Healthy=0, Moderate=1, Unhealthy=2
    diet_mapping = {'Healthy': 0, 'Moderate': 1, 'Unhealthy': 2}
    # Suicidal thoughts and family history: Yes=1, No=0
    binary_mapping = {'Yes': 1, 'No': 0}
    
    return [
        diet_mapping[diet],
        binary_mapping[suicidal],
        binary_mapping[family_hist]
    ]

# Prepare features in the correct order
features = np.array([[
    age,
    academic_pressure,
    study_satisfaction,
    *encode_features(dietary_habits, suicidal_thoughts, family_history),
    # study_hours,
    financial_stress
]])

# Make predictions
if st.button('Predict Depression Risk'):
    try:
        # Get predictions from all models
        lr_pred = models['lr'].predict(features)[0]
        nb_pred = models['nb'].predict(features)[0]
        rf_pred = models['rf'].predict(features)[0]
        svm_pred = models['svm'].predict(features)[0]
        ann_pred = (models['ann'].predict(features) > 0.5).astype(int)[0][0]

        # Collect votes
        votes = [lr_pred, nb_pred, rf_pred, svm_pred, ann_pred]
        prediction = 1 if sum(votes) >= 3 else 0  # Majority voting

        # Display results
        st.subheader('Prediction Results')
        st.write("Individual Model Predictions:")
        
        results = {
            'Logistic Regression': 'At Risk' if lr_pred == 1 else 'No Risk',
            'Naive Bayes': 'At Risk' if nb_pred == 1 else 'No Risk',
            'Random Forest': 'At Risk' if rf_pred == 1 else 'No Risk',
            'SVM': 'At Risk' if svm_pred == 1 else 'No Risk',
            'Neural Network': 'At Risk' if ann_pred == 1 else 'No Risk'
        }
        
        for model, result in results.items():
            st.write(f"{model}: {result}")
            
        st.subheader('Final Consensus Prediction')
        st.success('At Risk of Depression' if prediction == 1 else 'No Significant Risk of Depression')
        
        # Show voting breakdown
        st.write("\nVoting Breakdown:")
        st.write(f"Models predicting 'At Risk': {sum(votes)}")
        st.write(f"Models predicting 'No Risk': {len(votes) - sum(votes)}")
            
    except Exception as e:
        st.error(f'Error making prediction: {str(e)}')