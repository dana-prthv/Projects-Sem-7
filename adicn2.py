# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)

st.set_page_config(page_title="LTE Anomaly Detection", layout="wide")
st.title("📶 LTE Cellular Network Anomaly Detection")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload LTE dataset CSV", type="csv")

if uploaded_file is not None:
    try:
        # Try utf-8-sig first, then fallback to ISO-8859-1
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

        # Clean column names
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
        df.rename(columns=lambda x: x.replace("ï»¿Time", "Time"), inplace=True)

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        df = df.drop_duplicates()

        # -----------------------------
        # Preprocessing
        # -----------------------------
        st.subheader("Preprocessing Data")

        # Convert Time to numeric
        if "Time" in df.columns:
            df["TimeValue"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour + \
                              pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.minute / 60.0
            df.drop(columns=["Time"], inplace=True)

        # Numeric conversion
        if "maxUE_UL+DL" in df.columns:
            df["maxUE_UL+DL"] = pd.to_numeric(df["maxUE_UL+DL"], errors="coerce")

        # Encode CellName
        if "CellName" in df.columns:
            le = LabelEncoder()
            df["CellName"] = le.fit_transform(df["CellName"])

        df.dropna(inplace=True)

        if "Unusual" not in df.columns:
            st.error("❌ Target column 'Unusual' not found!")
            st.stop()

        X = df.drop(columns=["Unusual"])
        y = df["Unusual"]

        # Drop remaining non-numeric
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            st.warning(f"⚠️ Non-numeric columns found and dropped: {non_numeric_cols}")
            X = X.drop(columns=non_numeric_cols)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        st.success("✅ Data preprocessing completed successfully!")

        # -----------------------------
        # Sidebar: Model Selection
        # -----------------------------
        st.sidebar.title("Model & Prediction Settings")
        model_choice = st.sidebar.selectbox(
            "Choose a model",
            ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Logistic Regression"]
        )
        train_button = st.sidebar.button("Train Model")

        # -----------------------------
        # Train model function
        # -----------------------------
        @st.cache_data
        def train_model(choice):
            if choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif choice == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif choice == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            elif choice == "LightGBM":
                model = LGBMClassifier(n_estimators=100, random_state=42)
            elif choice == "CatBoost":
                model = CatBoostClassifier(verbose=0, random_state=42)
            else:
                model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            return model

        # -----------------------------
        # Train / Load Model
        # -----------------------------
        if train_button or "model" in st.session_state:
            if train_button:
                model = train_model(model_choice)
                st.session_state["model"] = model
            else:
                model = st.session_state["model"]

            st.success(f"✅ Model '{model_choice}' is ready!")

            # -----------------------------
            # Evaluation
            # -----------------------------
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            conf_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            prec, rec, _ = precision_recall_curve(y_test, y_prob)

            st.subheader("📊 Evaluation Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{accuracy:.3f}")
            c2.metric("Precision", f"{precision:.3f}")
            c3.metric("Recall", f"{recall:.3f}")
            c4.metric("F1 Score", f"{f1:.3f}")
            c5.metric("ROC-AUC", f"{roc_auc:.3f}")

            # -----------------------------
            # Plots
            # -----------------------------
            st.subheader("Evaluation Plots")
            # Use smaller subplots to fit nicely
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Confusion Matrix
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0,0])
            axes[0,0].set_title("Confusion Matrix")

            # ROC Curve
            axes[0,1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            axes[0,1].plot([0,1],[0,1],'k--')
            axes[0,1].set_title("ROC Curve")
            axes[0,1].set_xlabel("False Positive Rate")
            axes[0,1].set_ylabel("True Positive Rate")
            axes[0,1].legend()

            # Precision-Recall Curve
            axes[1,0].plot(rec, prec)
            axes[1,0].set_title("Precision-Recall Curve")
            axes[1,0].set_xlabel("Recall")
            axes[1,0].set_ylabel("Precision")

            # Feature Importance (only for tree-based models)
            axes[1,1].axis('off')
            if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]:
                importance = model.feature_importances_
                sns.barplot(x=X.columns, y=importance, ax=axes[1,1])
                axes[1,1].set_title("Feature Importance")
                axes[1,1].tick_params(axis='x', rotation=45)

            st.pyplot(fig)

            # -----------------------------
            # Predict on Custom Input
            # -----------------------------
            st.header("🔮 Predict on Custom Input")
            with st.form("predict_form"):
                user_input = {}
                for col in X.columns:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val, step=0.01)
                submitted = st.form_submit_button("Predict")

            if submitted:
                try:
                    input_df = pd.DataFrame([user_input])
                    input_scaled = scaler.transform(input_df)
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

                    if pred == 1:
                        st.error("🚨 Prediction: **Unusual** (Anomaly detected)")
                    else:
                        st.success("✅ Prediction: **Usual** (Normal behavior)")

                    if prob is not None:
                        st.info(f"Prediction Confidence: {prob:.3f}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("📂 Upload your LTE dataset CSV to start.")
