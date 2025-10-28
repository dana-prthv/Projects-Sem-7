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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)

# -----------------------------
# 🎨 Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="LTE Anomaly Detection", layout="wide")
st.title("📶 LTE Cellular Network Anomaly Detection")

uploaded_file = st.file_uploader("Upload LTE dataset CSV", type="csv")

if uploaded_file is not None:
    try:
        # Try decoding the CSV
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

        # Fix BOM and trim column names
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
        df.columns = [c.encode('utf-8').decode('utf-8', 'ignore') for c in df.columns]
        df.rename(columns=lambda x: x.replace("ï»¿Time", "Time"), inplace=True)

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        df = df.drop_duplicates()

        # -----------------------------
        # 🔧 Data Preprocessing
        # -----------------------------
        st.subheader("Preprocessing Data")

        # Convert Time -> numeric hour value
        if "Time" in df.columns:
            df["TimeValue"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour + \
                              pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.minute / 60.0
            df.drop(columns=["Time"], inplace=True)

        # Convert maxUE_UL+DL to numeric
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

        # Drop any remaining non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            st.warning(f"⚠️ Non-numeric columns found: {non_numeric_cols}. Dropping them.")
            X = X.drop(columns=non_numeric_cols)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        st.success("✅ Data preprocessing completed successfully!")

        # -----------------------------
        # 🧠 Model Selection
        # -----------------------------
        st.sidebar.title("Model & Prediction Settings")
        model_choice = st.sidebar.selectbox(
            "Choose a model",
            ["Random Forest", "Gradient Boosting", "XGBoost", "Logistic Regression"]
        )

        @st.cache_data
        def train_model(choice):
            if choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif choice == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif choice == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            else:
                model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            return model

        model = train_model(model_choice)

        # -----------------------------
        # 📊 Model Evaluation
        # -----------------------------
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        conf_matrix = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob) if y_prob is not None else ([], [], [])
        prec, rec, _ = precision_recall_curve(y_test, y_prob) if y_prob is not None else ([], [], [])

        st.subheader("📊 Evaluation Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{accuracy:.3f}")
        c2.metric("Precision", f"{precision:.3f}")
        c3.metric("Recall", f"{recall:.3f}")
        c4.metric("F1 Score", f"{f1:.3f}")
        c5.metric("ROC-AUC", f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "N/A")

        # -----------------------------
        # 📈 Visualization Layout
        # -----------------------------
        st.subheader("📉 Model Performance Visualization")

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax1)
            ax1.set_title("Confusion Matrix")
            st.pyplot(fig1)

        with col2:
            if y_prob is not None:
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax2.plot([0, 1], [0, 1], 'k--')
                ax2.set_title("ROC Curve")
                ax2.legend()
                st.pyplot(fig2)

        col3, col4 = st.columns(2)
        with col3:
            if y_prob is not None:
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                ax3.plot(rec, prec)
                ax3.set_xlabel("Recall")
                ax3.set_ylabel("Precision")
                ax3.set_title("Precision-Recall Curve")
                st.pyplot(fig3)

        with col4:
            if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                st.subheader("Feature Importance")
                importance = model.feature_importances_
                fig4, ax4 = plt.subplots(figsize=(4, 3))
                sns.barplot(x=X.columns, y=importance, ax=ax4)
                plt.xticks(rotation=45)
                ax4.set_title("Feature Importance")
                st.pyplot(fig4)

        # -----------------------------
        # 🧠 Custom Input Prediction
        # -----------------------------
        st.header("🔮 Predict on Custom Input")

        with st.form("predict_form"):
            st.write("Enter feature values for a new sample:")
            user_input = {}
            for col in X.columns:
                val = st.text_input(f"{col}", value=str(round(float(df[col].mean()), 3)))
                user_input[col] = float(val)
            submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                input_df = pd.DataFrame([user_input])
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

                if pred == 1:
                    st.error(f"🚨 Prediction: **Unusual** (Anomaly detected)")
                else:
                    st.success(f"✅ Prediction: **Usual** (Normal behavior)")

                if prob is not None:
                    st.write(f"Prediction Confidence: {prob:.3f}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("📂 Upload your LTE dataset CSV to start.")
