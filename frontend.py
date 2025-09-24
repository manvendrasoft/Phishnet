import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from sklearn.metrics import (
    roc_curve, auc, classification_report, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load models
ann_model = load_model("models/ann_model.h5")
rf_model = joblib.load("models/rf_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")

# Load scaler and PCA
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

# Streamlit config
st.set_page_config(page_title="PhishNet – Phishing Detector", layout="centered")

st.title("🛡️ PhishNet – Phishing Website Detector")
st.markdown("Detect if a website is **Legitimate ✅** or **Phishing 🚨** using ML & Deep Learning.")

# Model selector
model_choice = st.radio("🔍 Choose a model to predict with:", ["ANN", "Random Forest", "SVM"])

# Sample inputs
sample_phishing = [-1, 0, 1, 0, 1, 0, 0, -1, -1, 0, 1, 0, -1, 0, 1, -1, 1, 1, 0, 0, -1, 1, 0, -1, -1, 0, 1, 0, -1, 1]
sample_legit = [1, 1, 0, 0, -1, 0, 0, 1, 1, 0, -1, 0, 1, 0, -1, 1, -1, -1, 0, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

st.subheader("✍️ Enter Feature Values:")
feature_names = [f"Feature {i+1}" for i in range(30)]

preset_choice = st.radio("📋 Load Sample Data:", ["Manual Input", "Sample Phishing", "Sample Legitimate"])
if preset_choice == "Sample Phishing":
    user_input = sample_phishing
elif preset_choice == "Sample Legitimate":
    user_input = sample_legit
else:
    user_input = [st.number_input(label, step=0.1, key=i) for i, label in enumerate(feature_names)]

# Predict single input
if st.button("🔎 Predict"):
    try:
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_pca = pca.transform(input_scaled)

        if model_choice == "ANN":
            prob = ann_model.predict(input_pca)[0][0]
        elif model_choice == "Random Forest":
            prob = rf_model.predict_proba(input_pca)[0][1]
        else:
            prob = svm_model.predict_proba(input_pca)[0][1]

        label = "Phishing 🚨" if prob > 0.5 else "Legitimate ✅"
        st.success(f"🔐 Prediction: **{label}**")
        st.metric("Phishing Probability", f"{prob * 100:.2f} %")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# Upload CSV for batch prediction
st.subheader("📤 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV with 30 features (no Index/class)", type=["csv"])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)

        if df_uploaded.shape[1] != 30:
            st.error("❌ CSV must contain exactly 30 features.")
        else:
            X_scaled = scaler.transform(df_uploaded)
            X_pca = pca.transform(X_scaled)

            if model_choice == "ANN":
                probs = ann_model.predict(X_pca).flatten()
            elif model_choice == "Random Forest":
                probs = rf_model.predict_proba(X_pca)[:, 1]
            else:
                probs = svm_model.predict_proba(X_pca)[:, 1]

            labels = ["Phishing 🚨" if p > 0.5 else "Legitimate ✅" for p in probs]

            result_df = df_uploaded.copy()
            result_df["Probability"] = probs
            result_df["Prediction"] = labels

            st.success(f"✅ Predictions completed for {len(result_df)} rows.")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "phishing_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ File processing error: {e}")

# Load full dataset and split for evaluation
@st.cache_data
def load_data():
    df = pd.read_csv("phishing.csv")
    X = df.drop(['Index', 'class'], axis=1)
    y = df['class']
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return train_test_split(X_pca, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# ROC Curve visualization
st.subheader("📊 ROC Curve Preview")
if st.checkbox("Show ROC Curve for Test Set"):
    try:
        if model_choice == "ANN":
            probs = ann_model.predict(X_test).flatten()
        elif model_choice == "Random Forest":
            probs = rf_model.predict_proba(X_test)[:, 1]
        else:
            probs = svm_model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve – {model_choice}')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Could not generate ROC: {e}")

# Show Confusion Matrix & Classification Report
st.subheader("📈 Model Evaluation Metrics")
if st.checkbox("Show Confusion Matrix and Classification Report"):
    try:
        if model_choice == "ANN":
            probs = ann_model.predict(X_test).flatten()
            preds = probs > 0.5
        elif model_choice == "Random Forest":
            probs = rf_model.predict_proba(X_test)[:, 1]
            preds = probs > 0.5
        else:
            probs = svm_model.predict_proba(X_test)[:, 1]
            preds = probs > 0.5

        # Classification report
        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("### Classification Report")
        st.dataframe(report_df)

        # Confusion matrix plot
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, cmap='Blues', normalize='true')
        ax.set_title(f"{model_choice} Confusion Matrix (Normalized)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error displaying metrics: {e}")

# Final model comparison summary
st.subheader("📊 Final Model Performance Summary")
if st.checkbox("Show Summary Comparison for All Models"):
    try:
        # Predict all models on test set
        ann_probs = ann_model.predict(X_test).flatten()
        ann_pred = ann_probs > 0.5

        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        rf_pred = rf_probs > 0.5

        svm_probs = svm_model.predict_proba(X_test)[:, 1]
        svm_pred = svm_probs > 0.5

        # Calculate metrics
        models = ['ANN', 'Random Forest', 'SVM']

        auc_scores = [
            roc_auc_score(y_test, ann_probs),
            roc_auc_score(y_test, rf_probs),
            roc_auc_score(y_test, svm_probs)
        ]

        accuracy_scores = [
            accuracy_score(y_test, ann_pred) * 100,
            accuracy_score(y_test, rf_pred) * 100,
            accuracy_score(y_test, svm_pred) * 100
        ]

        precision_scores = [
            precision_score(y_test, ann_pred),
            precision_score(y_test, rf_pred),
            precision_score(y_test, svm_pred)
        ]

        recall_scores = [
            recall_score(y_test, ann_pred),
            recall_score(y_test, rf_pred),
            recall_score(y_test, svm_pred)
        ]

        f1_scores = [
            f1_score(y_test, ann_pred),
            f1_score(y_test, rf_pred),
            f1_score(y_test, svm_pred)
        ]

        summary_df = pd.DataFrame({
            'Model': models,
            'AUC Score': auc_scores,
            'Accuracy (%)': accuracy_scores,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1-Score': f1_scores
        })

        st.dataframe(summary_df.style.format({
            'AUC Score': '{:.3f}',
            'Accuracy (%)': '{:.2f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}'
        }))

        # Bar chart comparison
        import numpy as np
        x = np.arange(len(models))
        width = 0.13

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - 2*width, summary_df['AUC Score'] * 100, width, label='AUC Score (%)', color='mediumslateblue')
        bars2 = ax.bar(x - width, summary_df['Accuracy (%)'], width, label='Accuracy (%)', color='mediumseagreen')
        bars3 = ax.bar(x, summary_df['Precision'] * 100, width, label='Precision (%)', color='coral')
        bars4 = ax.bar(x + width, summary_df['Recall'] * 100, width, label='Recall (%)', color='gold')
        bars5 = ax.bar(x + 2*width, summary_df['F1-Score'] * 100, width, label='F1-Score (%)', color='orchid')

        # Add labels on bars
        for bars in [bars1, bars2, bars3, bars4, bars5]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model')
        ax.set_ylabel('Score (%)')
        ax.set_title('Model Comparison: Metrics Overview')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error generating summary comparison: {e}")
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from sklearn.metrics import (
    roc_curve, auc, classification_report, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load models
ann_model = load_model("models/ann_model.h5")
rf_model = joblib.load("models/rf_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")

# Load scaler and PCA
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

# Streamlit config
st.set_page_config(page_title="PhishNet – Phishing Detector", layout="centered")

st.title("🛡️ PhishNet – Phishing Website Detector")
st.markdown("Detect if a website is **Legitimate ✅** or **Phishing 🚨** using ML & Deep Learning.")

# Model selector
model_choice = st.radio("🔍 Choose a model to predict with:", ["ANN", "Random Forest", "SVM"])

# Sample inputs
sample_phishing = [-1, 0, 1, 0, 1, 0, 0, -1, -1, 0, 1, 0, -1, 0, 1, -1, 1, 1, 0, 0, -1, 1, 0, -1, -1, 0, 1, 0, -1, 1]
sample_legit = [1, 1, 0, 0, -1, 0, 0, 1, 1, 0, -1, 0, 1, 0, -1, 1, -1, -1, 0, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

st.subheader("✍️ Enter Feature Values:")
feature_names = [f"Feature {i+1}" for i in range(30)]

preset_choice = st.radio("📋 Load Sample Data:", ["Manual Input", "Sample Phishing", "Sample Legitimate"])
if preset_choice == "Sample Phishing":
    user_input = sample_phishing
elif preset_choice == "Sample Legitimate":
    user_input = sample_legit
else:
    user_input = [st.number_input(label, step=0.1, key=i) for i, label in enumerate(feature_names)]

# Predict single input
if st.button("🔎 Predict"):
    try:
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_pca = pca.transform(input_scaled)

        if model_choice == "ANN":
            prob = ann_model.predict(input_pca)[0][0]
        elif model_choice == "Random Forest":
            prob = rf_model.predict_proba(input_pca)[0][1]
        else:
            prob = svm_model.predict_proba(input_pca)[0][1]

        label = "Phishing 🚨" if prob > 0.5 else "Legitimate ✅"
        st.success(f"🔐 Prediction: **{label}**")
        st.metric("Phishing Probability", f"{prob * 100:.2f} %")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# Upload CSV for batch prediction
st.subheader("📤 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV with 30 features (no Index/class)", type=["csv"])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)

        if df_uploaded.shape[1] != 30:
            st.error("❌ CSV must contain exactly 30 features.")
        else:
            X_scaled = scaler.transform(df_uploaded)
            X_pca = pca.transform(X_scaled)

            if model_choice == "ANN":
                probs = ann_model.predict(X_pca).flatten()
            elif model_choice == "Random Forest":
                probs = rf_model.predict_proba(X_pca)[:, 1]
            else:
                probs = svm_model.predict_proba(X_pca)[:, 1]

            labels = ["Phishing 🚨" if p > 0.5 else "Legitimate ✅" for p in probs]

            result_df = df_uploaded.copy()
            result_df["Probability"] = probs
            result_df["Prediction"] = labels

            st.success(f"✅ Predictions completed for {len(result_df)} rows.")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "phishing_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ File processing error: {e}")

# Load full dataset and split for evaluation
@st.cache_data
def load_data():
    df = pd.read_csv("phishing.csv")
    X = df.drop(['Index', 'class'], axis=1)
    y = df['class']
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return train_test_split(X_pca, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# ROC Curve visualization
st.subheader("📊 ROC Curve Preview")
if st.checkbox("Show ROC Curve for Test Set"):
    try:
        if model_choice == "ANN":
            probs = ann_model.predict(X_test).flatten()
        elif model_choice == "Random Forest":
            probs = rf_model.predict_proba(X_test)[:, 1]
        else:
            probs = svm_model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve – {model_choice}')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Could not generate ROC: {e}")

# Show Confusion Matrix & Classification Report
st.subheader("📈 Model Evaluation Metrics")
if st.checkbox("Show Confusion Matrix and Classification Report"):
    try:
        if model_choice == "ANN":
            probs = ann_model.predict(X_test).flatten()
            preds = probs > 0.5
        elif model_choice == "Random Forest":
            probs = rf_model.predict_proba(X_test)[:, 1]
            preds = probs > 0.5
        else:
            probs = svm_model.predict_proba(X_test)[:, 1]
            preds = probs > 0.5

        # Classification report
        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("### Classification Report")
        st.dataframe(report_df)

        # Confusion matrix plot
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, cmap='Blues', normalize='true')
        ax.set_title(f"{model_choice} Confusion Matrix (Normalized)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error displaying metrics: {e}")

# Final model comparison summary
st.subheader("📊 Final Model Performance Summary")
if st.checkbox("Show Summary Comparison for All Models"):
    try:
        # Predict all models on test set
        ann_probs = ann_model.predict(X_test).flatten()
        ann_pred = ann_probs > 0.5

        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        rf_pred = rf_probs > 0.5

        svm_probs = svm_model.predict_proba(X_test)[:, 1]
        svm_pred = svm_probs > 0.5

        # Calculate metrics
        models = ['ANN', 'Random Forest', 'SVM']

        auc_scores = [
            roc_auc_score(y_test, ann_probs),
            roc_auc_score(y_test, rf_probs),
            roc_auc_score(y_test, svm_probs)
        ]

        accuracy_scores = [
            accuracy_score(y_test, ann_pred) * 100,
            accuracy_score(y_test, rf_pred) * 100,
            accuracy_score(y_test, svm_pred) * 100
        ]

        precision_scores = [
            precision_score(y_test, ann_pred),
            precision_score(y_test, rf_pred),
            precision_score(y_test, svm_pred)
        ]

        recall_scores = [
            recall_score(y_test, ann_pred),
            recall_score(y_test, rf_pred),
            recall_score(y_test, svm_pred)
        ]

        f1_scores = [
            f1_score(y_test, ann_pred),
            f1_score(y_test, rf_pred),
            f1_score(y_test, svm_pred)
        ]

        summary_df = pd.DataFrame({
            'Model': models,
            'AUC Score': auc_scores,
            'Accuracy (%)': accuracy_scores,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1-Score': f1_scores
        })

        st.dataframe(summary_df.style.format({
            'AUC Score': '{:.3f}',
            'Accuracy (%)': '{:.2f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}'
        }))

        # Bar chart comparison
        import numpy as np
        x = np.arange(len(models))
        width = 0.13

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - 2*width, summary_df['AUC Score'] * 100, width, label='AUC Score (%)', color='mediumslateblue')
        bars2 = ax.bar(x - width, summary_df['Accuracy (%)'], width, label='Accuracy (%)', color='mediumseagreen')
        bars3 = ax.bar(x, summary_df['Precision'] * 100, width, label='Precision (%)', color='coral')
        bars4 = ax.bar(x + width, summary_df['Recall'] * 100, width, label='Recall (%)', color='gold')
        bars5 = ax.bar(x + 2*width, summary_df['F1-Score'] * 100, width, label='F1-Score (%)', color='orchid')

        # Add labels on bars
        for bars in [bars1, bars2, bars3, bars4, bars5]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model')
        ax.set_ylabel('Score (%)')
        ax.set_title('Model Comparison: Metrics Overview')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error generating summary comparison: {e}")
