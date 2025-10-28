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
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)

# -------------------------------------
# Streamlit App Configuration
# -------------------------------------
st.set_page_config(page_title="LTE Anomaly Detection", layout="wide")
st.title("📶 LTE Cellular Network Anomaly Detection & Root Cause Analysis")

# -------------------------------------
# File Upload
# -------------------------------------
uploaded_file = st.file_uploader("Upload LTE dataset CSV", type="csv")

if uploaded_file is not None:
    try:
        # Read CSV safely
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
        df.rename(columns=lambda x: x.replace("ï»¿Time", "Time"), inplace=True)

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        df = df.drop_duplicates()

        # -------------------------------------
        # Data Preprocessing
        # -------------------------------------
        st.subheader("Preprocessing Data")

        if "Time" in df.columns:
            df["TimeValue"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour + \
                              pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.minute / 60.0
            df.drop(columns=["Time"], inplace=True)

        if "maxUE_UL+DL" in df.columns:
            df["maxUE_UL+DL"] = pd.to_numeric(df["maxUE_UL+DL"], errors="coerce")

        if "CellName" in df.columns:
            le = LabelEncoder()
            df["CellName"] = le.fit_transform(df["CellName"])

        df.dropna(inplace=True)

        if "Unusual" not in df.columns:
            st.error("❌ Target column 'Unusual' not found!")
            st.stop()

        X = df.drop(columns=["Unusual"])
        y = df["Unusual"]

        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            st.warning(f"⚠️ Non-numeric columns dropped: {non_numeric_cols}")
            X = X.drop(columns=non_numeric_cols)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        st.success("✅ Data preprocessing completed successfully!")

        # -------------------------------------
        # Sidebar Settings
        # -------------------------------------
        st.sidebar.title("Model & Prediction Settings")
        model_choice = st.sidebar.selectbox(
            "Choose a model",
            ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Logistic Regression", "MLP Classifier"]
        )
        train_button = st.sidebar.button("Train Model")

        # -------------------------------------
        # Train Model Function
        # -------------------------------------
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
            elif choice == "MLP Classifier":
                model = MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=500,
                    early_stopping=True,
                    random_state=42
                )
            else:
                model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            return model

        # -------------------------------------
        # Model Training / Evaluation
        # -------------------------------------
        if train_button or "model" in st.session_state:
            if train_button:
                model = train_model(model_choice)
                st.session_state["model"] = model
            else:
                model = st.session_state["model"]

            st.success(f"✅ Model '{model_choice}' is ready!")

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

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

            # -------------------------------------
            # Evaluation Plots
            # -------------------------------------
            st.subheader("Evaluation Plots")
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
            axes[0, 0].set_title("Confusion Matrix")

            axes[0, 1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_title("ROC Curve")
            axes[0, 1].legend()

            axes[1, 0].plot(rec, prec)
            axes[1, 0].set_title("Precision-Recall Curve")

            axes[1, 1].axis('off')
            if hasattr(model, "feature_importances_"):
                sns.barplot(x=X.columns, y=model.feature_importances_, ax=axes[1, 1])
                axes[1, 1].set_title("Feature Importance")
                axes[1, 1].tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            # -------------------------------------
            # 🧠 Root Cause Analysis Section
            # -------------------------------------
            st.header("🧩 Root Cause Analysis (KPI Deviations)")

            normal_samples = df[df["Unusual"] == 0][X.columns]
            anomaly_samples = df[df["Unusual"] == 1][X.columns]

            normal_means = normal_samples.mean()
            anomaly_means = anomaly_samples.mean()
            kpi_diff = pd.DataFrame({
                "KPI": X.columns,
                "Normal_Mean": normal_means,
                "Anomaly_Mean": anomaly_means
            })
            kpi_diff["Percent_Change"] = 100 * abs(kpi_diff["Anomaly_Mean"] - kpi_diff["Normal_Mean"]) / (abs(kpi_diff["Normal_Mean"]) + 1e-6)
            kpi_diff = kpi_diff.sort_values("Percent_Change", ascending=False)

            st.write("### Top Deviating KPIs During Anomalies")
            st.dataframe(kpi_diff.head(10))

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="KPI", y="Percent_Change", data=kpi_diff.head(10), ax=ax)
            ax.set_title("Top KPI Deviations")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            # Generate text summary
            summary_lines = []
            for _, row in kpi_diff.head(5).iterrows():
                direction = "increased" if row["Anomaly_Mean"] > row["Normal_Mean"] else "decreased"
                summary_lines.append(f"- **{row['KPI']}** {direction} by {row['Percent_Change']:.1f}% during anomalies.")
            st.markdown("### 🧠 Root Cause Summary")
            st.markdown("\n".join(summary_lines))

            # -------------------------------------
            # 🌀 Clustering Anomalies (DBSCAN)
            # -------------------------------------
            st.header("🌀 Anomaly Clustering (Pattern Discovery)")

            anomaly_scaled = scaler.transform(anomaly_samples)
            dbscan = DBSCAN(eps=1.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(anomaly_scaled)
            valid_clusters = np.unique(cluster_labels[cluster_labels != -1])

            if len(valid_clusters) == 0:
                st.warning("No clear clusters found — anomalies may be random or too similar.")
            else:
                st.success(f"✅ Found {len(valid_clusters)} distinct anomaly types.")
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(anomaly_scaled)

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=cluster_labels, palette="tab10", ax=ax)
                ax.set_title("Anomaly Clusters (DBSCAN + PCA)")
                st.pyplot(fig)

                anomaly_samples["Cluster"] = cluster_labels
                cluster_summary = anomaly_samples.groupby("Cluster").mean().round(3)

                st.subheader("📊 Cluster KPI Summary")
                st.dataframe(cluster_summary)

                # Generate textual summaries per cluster
                normal_means = normal_samples.mean()
                st.markdown("### 🧩 Cluster Interpretations")
                for cluster in valid_clusters:
                    cluster_means = cluster_summary.loc[cluster]
                    diff = abs(cluster_means - normal_means)
                    top_kpis = diff.sort_values(ascending=False).head(3).index.tolist()
                    st.write(f"**Cluster {cluster}:** Dominated by changes in {', '.join(top_kpis)}.")
                    if any("latency" in k.lower() for k in top_kpis):
                        st.caption("→ Likely congestion or interference.")
                    elif any("throughput" in k.lower() for k in top_kpis):
                        st.caption("→ Possible capacity or bandwidth issue.")
                    elif any("failure" in k.lower() for k in top_kpis):
                        st.caption("→ Mobility or signaling issue.")
                    else:
                        st.caption("→ General anomaly pattern (needs further inspection).")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("📂 Upload your LTE dataset CSV to start.")
