import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from pytorch_tabnet.tab_model import TabNetClassifier
from features import extract_features  # Import your feature extraction module

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="DeepGuard", 
                   page_icon="🛡️",
                   layout="wide")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------
# Title & Tagline
# -----------------------------------
st.title("DeepGuard: Hybrid Ransomware Detection System")
st.markdown("<h3 style='color: #2e6c80;'>Trained with EMBER dataset</h3>", unsafe_allow_html=True)
st.markdown("""
This system uses a combination of classical machine learning and deep learning models for malware detection.  
Select your preferred model, upload a file, and view the predicted result.
""")
st.markdown("---")

# -----------------------------------
# Model Selection & Evaluation Metrics (Displayed in an Expander After Selection)
# -----------------------------------
st.header("Select Model for Prediction")
model_options = ["Random Forest", "XGBoost", "Baseline Deep Learning", "TabNet", "Ensemble"]
selected_model = st.selectbox("Choose the model", model_options)

def update_metrics(model_name):
    if model_name == "Random Forest":
        accuracy, precision, recall, f1_score = "91%", "91%", "91%", "91%"
    elif model_name == "XGBoost":
        accuracy, precision, recall, f1_score = "86%", "86%", "86%", "86%"
    elif model_name == "Baseline Deep Learning":
        accuracy, precision, recall, f1_score = "62%", "70%", "61%", "57%"
    elif model_name == "TabNet":
        accuracy, precision, recall, f1_score = "50%", "75%", "50%", "33%"
    elif model_name == "Ensemble":
        accuracy, precision, recall, f1_score = "90%", "90%", "90%", "90%"
    else:
        accuracy, precision, recall, f1_score = "N/A", "N/A", "N/A", "N/A"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", accuracy)
    col2.metric("Precision", precision)
    col3.metric("Recall", recall)
    col4.metric("F1-Score", f1_score)

with st.expander("Show Evaluation Metrics for Selected Model"):
    update_metrics(selected_model)

st.markdown("---")

# -----------------------------------
# File Upload & Detection History Section
# -----------------------------------
st.header("File Upload & Detection History")

# Initialize session state for upload history if not present.
if 'upload_history' not in st.session_state:
    st.session_state['upload_history'] = []

uploaded_file = st.file_uploader("Upload a file (exe, dll, bin, txt)", type=["exe", "dll", "bin", "txt"])

if st.button("Run Detection") and uploaded_file is not None:
    # Save uploaded file temporarily.
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)
    temp_file_path = os.path.join(temp_folder, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features
    features = extract_features(temp_file_path)
    if features is None:
        st.error("Feature extraction failed. Please try another file.")
    else:
        model_folder = os.path.join("..", "Models")  # Adjust relative path as necessary.
        outcome = ""
        if selected_model == "Random Forest":
            model_path = os.path.join(model_folder, "rf_model.pkl")
            model = joblib.load(model_path)
            pred = model.predict(features)
            outcome = "MALICIOUS" if pred[0] == 1 else "BENIGN"
        elif selected_model == "XGBoost":
            model_path = os.path.join(model_folder, "xgb_model.pkl")
            model = joblib.load(model_path)
            pred = model.predict(features)
            outcome = "MALICIOUS" if pred[0] == 1 else "BENIGN"
        elif selected_model == "Baseline Deep Learning":
            model_path = os.path.join(model_folder, "baseline_deep_model.h5")
            model = load_model(model_path)
            pred_prob = model.predict(features).ravel()[0]
            outcome = "MALICIOUS" if pred_prob > 0.5 else "BENIGN"
            outcome += f" (Prob: {pred_prob:.2f})"
        elif selected_model == "TabNet":
            model_path = os.path.join(model_folder, "tabnet_model.zip")
            model = TabNetClassifier()
            model.load_model(model_path)
            pred_prob = model.predict_proba(features.astype(np.float32))[:, 1][0]
            outcome = "MALICIOUS" if pred_prob > 0.5 else "BENIGN"
            outcome += f" (Prob: {pred_prob:.2f})"
        elif selected_model == "Ensemble":
            rf_model = joblib.load(os.path.join(model_folder, "rf_model.pkl"))
            xgb_model = joblib.load(os.path.join(model_folder, "xgb_model.pkl"))
            dl_model = load_model(os.path.join(model_folder, "baseline_deep_model.h5"))
            tabnet_model = TabNetClassifier()
            tabnet_model.load_model(os.path.join(model_folder, "tabnet_model.zip"))
            
            rf_prob = rf_model.predict_proba(features)[:, 1][0]
            xgb_prob = xgb_model.predict_proba(features)[:, 1][0]
            dl_prob = dl_model.predict(features).ravel()[0]
            tabnet_prob = tabnet_model.predict_proba(features.astype(np.float32))[:, 1][0]
            
            ensemble_prob = (rf_prob + xgb_prob + dl_prob + tabnet_prob) / 4.0
            outcome = "MALICIOUS" if ensemble_prob > 0.5 else "BENIGN"
            outcome += f" (Prob: {ensemble_prob:.2f})"
        else:
            outcome = "Model not found."
        
        # Append details to session state history, including model used.
        st.session_state['upload_history'].append({
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize (bytes)": uploaded_file.size,
            "Model Used": selected_model,
            "Prediction": outcome
        })
        st.success("Prediction complete!")
        # Optionally remove the temporary file: os.remove(temp_file_path)

# Display the upload history table.
st.subheader("Upload History & Detection Results")
if st.session_state['upload_history']:
    history_df = pd.DataFrame(st.session_state['upload_history'])
    st.table(history_df)
else:
    st.write("No files have been uploaded yet.")

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.markdown("""
**DeepGuard Note:**  
This system is trained on the EMBER dataset using a hybrid approach that includes classical machine learning and deep learning techniques.
Future enhancements include further deep model tuning, advanced feature extraction, and improved UI/UX.
""")
