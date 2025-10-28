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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict

st.set_page_config(page_title="LTE Anomaly Detection with Root Cause Analysis", layout="wide")
st.title("📶 LTE Cellular Network Anomaly Detection — Root Cause Analysis & Clustering")

# -----------------------------
# Helper functions (caching where appropriate)
# -----------------------------
@st.cache_data
def safe_read_csv(uploaded_file):
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    except Exception as e:
        raise e
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
    df.rename(columns=lambda x: x.replace("ï»¿Time", "Time"), inplace=True)
    return df

@st.cache_data
def get_trained_model(choice, X_train, y_train):
    """Train chosen model (cached)."""
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

def root_cause_for_sample(sample_scaled: np.ndarray,
                          X_train_scaled: np.ndarray,
                          y_train: pd.Series,
                          X_train_df: pd.DataFrame,
                          feature_names: list,
                          k_neighbors: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a single sample (1d scaled vector), find nearest k normal training samples,
    compute mean normal KPI vector, compute differences and percent changes, return
    two DataFrames: (sorted absolute change table) and (top changes for display).
    """
    # Build nearest neighbors using only normal samples in train (y_train == 0)
    normal_idx = np.where(y_train == 0)[0]
    if len(normal_idx) == 0:
        raise ValueError("No normal samples found in training set to compare against.")

    X_normals = X_train_scaled[normal_idx]

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X_normals)), metric="euclidean")
    nn.fit(X_normals)
    dists, idxs = nn.kneighbors(sample_scaled.reshape(1, -1), return_distance=True)
    nearest_normals = X_normals[idxs[0]]  # shape (k, n_features)

    mean_normal = nearest_normals.mean(axis=0)  # mean KPI vector
    diff = sample_scaled.flatten() - mean_normal
    abs_diff = np.abs(diff)
    # For percent change, avoid divide by zero by using mean_normal + tiny epsilon
    eps = 1e-9
    percent_change = diff / (np.where(np.abs(mean_normal) < eps, eps, mean_normal)) * 100.0

    table = pd.DataFrame({
        "feature": feature_names,
        "mean_normal": mean_normal,
        "sample_value": sample_scaled.flatten(),
        "diff": diff,
        "abs_diff": abs_diff,
        "percent_change": percent_change
    })
    table_sorted = table.sort_values("abs_diff", ascending=False).reset_index(drop=True)

    # Display top 5
    top5 = table_sorted.head(5).copy()
    # For readability, inverse-transform isn't possible here since we don't track per-feature original scale,
    # so these numbers are in scaled units; we'll communicate that in UI. If we want original-scale numbers,
    # pass original sample & normal mean in unscaled form too.
    return table_sorted, top5

@st.cache_data
def anomaly_clustering(X_anomaly_scaled: np.ndarray,
                       eps: float = 0.5,
                       min_samples: int = 5) -> Tuple[np.ndarray, DBSCAN]:
    """
    Run DBSCAN on scaled anomaly vectors and return cluster labels and model.
    """
    if X_anomaly_scaled.shape[0] == 0:
        return np.array([]), None
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X_anomaly_scaled)
    return labels, db

def cluster_representatives(X_scaled: np.ndarray, labels: np.ndarray, feature_names: list, original_df: pd.DataFrame, n_rep: int = 3):
    """
    For each cluster (label >= 0), compute centroid and find nearest n_rep samples in original_df.
    Returns dict: label -> dict(size, centroid, representatives DataFrame)
    """
    reps = {}
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        if lab == -1:
            continue
        indices = np.where(labels == lab)[0]
        cluster_points = X_scaled[indices]
        centroid = cluster_points.mean(axis=0)
        # nearest to centroid
        dists = np.linalg.norm(cluster_points - centroid.reshape(1, -1), axis=1)
        order = np.argsort(dists)[:min(n_rep, len(indices))]
        rep_indices = indices[order]
        reps[lab] = {
            "size": len(indices),
            "centroid": centroid,
            "representatives": original_df.iloc[rep_indices].copy()
        }
    return reps

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload LTE dataset CSV", type="csv")

if uploaded_file is not None:
    try:
        df = safe_read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        df = df.drop_duplicates()

        # -----------------------------
        # Basic preprocessing
        # -----------------------------
        st.subheader("Preprocessing Data")
        # Convert Time column if present
        if "Time" in df.columns:
            df["TimeValue"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour + \
                              pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.minute / 60.0
            df.drop(columns=["Time"], inplace=True)

        # Try numeric conversion for common columns
        if "maxUE_UL+DL" in df.columns:
            df["maxUE_UL+DL"] = pd.to_numeric(df["maxUE_UL+DL"], errors="coerce")

        # Encode CellName if present
        if "CellName" in df.columns:
            le = LabelEncoder()
            df["CellName"] = le.fit_transform(df["CellName"])

        df.dropna(inplace=True)

        if "Unusual" not in df.columns:
            st.error("❌ Target column 'Unusual' not found! Please include a binary target column named 'Unusual' (0 = normal, 1 = anomaly).")
            st.stop()

        # Split X and y
        X = df.drop(columns=["Unusual"])
        y = df["Unusual"].astype(int)

        # Drop non-numeric columns and warn
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            st.warning(f"⚠️ Non-numeric columns found and dropped: {non_numeric_cols}")
            X = X.drop(columns=non_numeric_cols)

        feature_names = list(X.columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        st.success("✅ Data preprocessing completed successfully!")

        # -----------------------------
        # Sidebar: Model Selection
        # -----------------------------
        st.sidebar.title("Model & Prediction Settings")
        model_choice = st.sidebar.selectbox(
            "Choose a model",
            ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Logistic Regression", "MLP Classifier"]
        )
        train_button = st.sidebar.button("Train Model")

        if train_button or "model" in st.session_state:
            if train_button:
                with st.spinner("Training model..."):
                    model = get_trained_model(model_choice, X_train_scaled, y_train)
                st.session_state["model"] = model
            else:
                model = st.session_state["model"]

            st.success(f"✅ Model '{model_choice}' is ready!")

            # -----------------------------
            # Evaluation
            # -----------------------------
            y_pred = model.predict(X_test_scaled)
            # prediction probability or decision func
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                try:
                    y_prob = model.decision_function(X_test_scaled)
                except Exception:
                    # fallback
                    y_prob = np.zeros_like(y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            try:
                roc_auc = roc_auc_score(y_test, y_prob)
            except Exception:
                roc_auc = float("nan")
            conf_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob) if len(np.unique(y_test)) > 1 else ([], [], [])
            prec, rec, _ = precision_recall_curve(y_test, y_prob) if len(np.unique(y_test)) > 1 else ([], [], [])

            st.subheader("📊 Evaluation Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{accuracy:.3f}")
            c2.metric("Precision", f"{precision:.3f}")
            c3.metric("Recall", f"{recall:.3f}")
            c4.metric("F1 Score", f"{f1:.3f}")
            c5.metric("ROC-AUC", f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "N/A")

            # -----------------------------
            # Plots: Confusion, ROC, PR
            # -----------------------------
            st.subheader("Evaluation Plots")
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
            axes[0, 0].set_title("Confusion Matrix")

            if len(fpr) > 0:
                axes[0, 1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                axes[0, 1].plot([0, 1], [0, 1], 'k--')
                axes[0, 1].set_title("ROC Curve")
                axes[0, 1].set_xlabel("False Positive Rate")
                axes[0, 1].set_ylabel("True Positive Rate")
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, "ROC not available", ha='center')

            if len(rec) > 0:
                axes[1, 0].plot(rec, prec)
                axes[1, 0].set_title("Precision-Recall Curve")
                axes[1, 0].set_xlabel("Recall")
                axes[1, 0].set_ylabel("Precision")
            else:
                axes[1, 0].text(0.5, 0.5, "PR not available", ha='center')

            # Feature importance / coefficients
            axes[1, 1].axis('off')
            try:
                if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]:
                    importance = model.feature_importances_
                    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
                    sns.barplot(x="importance", y="feature", data=fi_df.head(10), ax=axes[1, 1])
                    axes[1, 1].set_title("Top Feature Importances")
                elif model_choice in ["Logistic Regression", "MLP Classifier"]:
                    if hasattr(model, "coef_"):
                        coef = np.abs(model.coef_[0])
                        coef_df = pd.DataFrame({"feature": feature_names, "coef": coef}).sort_values("coef", ascending=False)
                        sns.barplot(x="coef", y="feature", data=coef_df.head(10), ax=axes[1, 1])
                        axes[1, 1].set_title("Top Feature Coefficients (abs)")
            except Exception:
                pass

            st.pyplot(fig)

            # -----------------------------
            # ROOT CAUSE ANALYSIS UI
            # -----------------------------
            st.header("🔎 Root Cause Analysis (per-sample)")

            st.markdown("""
            Select a sample to analyze its root causes. The app will compare the selected sample to the nearest normal samples from the training set,
            compute which KPIs differ most (absolute and percent), and show the top contributors.
            """)

            # Choose a sample from the test set (index)
            test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
            test_df_orig = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=feature_names) if hasattr(scaler, "mean_") else test_df
            test_df["Unusual_true"] = list(y_test.values)
            test_df["Pred"] = list(y_pred)
            st.write("Test set preview (scaled features + labels)")
            st.dataframe(test_df.head(10))

            sample_idx = st.number_input("Select test sample index (row number in test set above)", min_value=0, max_value=len(test_df)-1, value=0, step=1)
            analyze_button = st.button("Run Root Cause on selected sample")

            if analyze_button:
                sample_scaled = X_test_scaled[sample_idx]
                sample_true = y_test.values[sample_idx]
                sample_pred = y_pred[sample_idx]

                st.info(f"Selected sample: true label = {sample_true}, model prediction = {sample_pred}")

                try:
                    full_table, top5 = root_cause_for_sample(
                        sample_scaled=sample_scaled,
                        X_train_scaled=X_train_scaled,
                        y_train=y_train.values,
                        X_train_df=pd.DataFrame(scaler.inverse_transform(X_train_scaled), columns=feature_names) if hasattr(scaler, "mean_") else pd.DataFrame(X_train_scaled, columns=feature_names),
                        feature_names=feature_names,
                        k_neighbors=20
                    )

                    st.subheader("Top feature differences (scaled units)")
                    st.table(top5[["feature", "mean_normal", "sample_value", "diff", "percent_change"]].assign(
                        percent_change=lambda df_: df_["percent_change"].map(lambda v: f"{v:.1f}%")
                    ))

                    # bar plot of top6 diffs
                    fig2, ax2 = plt.subplots(figsize=(8,4))
                    sns.barplot(x="diff", y="feature", data=full_table.head(10), ax=ax2)
                    ax2.set_title("Top 10 feature differences (sample - mean normal) [scaled units]")
                    st.pyplot(fig2)

                    # For extra clarity show full sorted table head
                    st.subheader("Full sorted feature difference table (top 20)")
                    st.dataframe(full_table.head(20).assign(percent_change=lambda df_: df_["percent_change"].map(lambda v: f"{v:.2f}%")))
                    st.success("Root cause analysis complete. Note: numbers are in scaled units (z-scores). To show original units, pass unscaled data.")
                except Exception as e:
                    st.error(f"Root cause analysis failed: {e}")

            # -----------------------------
            # ANOMALY CLUSTERING (global)
            # -----------------------------
            st.header("🧩 Anomaly Clustering (DBSCAN) — discover anomaly types")

            st.markdown("""
            This section clusters samples that the model labeled as anomalies (or ground-truth anomalies).
            - You can choose to cluster **predicted anomalies** (model says `1`) or **true anomalies** (`Unusual==1`).
            - DBSCAN is useful because it discovers clusters of arbitrary shape and marks noise (-1).
            """)

            cluster_mode = st.selectbox("Cluster mode", ["Predicted anomalies (model)","Ground-truth anomalies (dataset)"])
            eps = st.slider("DBSCAN eps (distance threshold)", min_value=0.1, max_value=5.0, value=0.8, step=0.1)
            min_samples = st.slider("DBSCAN min_samples", min_value=2, max_value=50, value=5, step=1)
            run_cluster = st.button("Run Clustering")

            if run_cluster:
                if cluster_mode == "Predicted anomalies (model)":
                    anomaly_mask = (y_pred == 1)
                    source_tag = "predicted"
                else:
                    anomaly_mask = (y_test.values == 1)
                    source_tag = "ground-truth"

                X_anom = X_test_scaled[anomaly_mask]
                if X_anom.shape[0] == 0:
                    st.warning(f"No anomaly samples to cluster in mode '{cluster_mode}'.")
                else:
                    labels, db = anomaly_clustering(X_anom, eps=eps, min_samples=min_samples)
                    st.write(f"Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters (excluding noise) and {np.sum(labels==-1)} noise points.")
                    # cluster sizes
                    unique, counts = np.unique(labels, return_counts=True)
                    cluster_stats = pd.DataFrame({"label": unique, "count": counts}).sort_values("label")
                    st.subheader("Cluster counts")
                    st.table(cluster_stats)

                    # PCA for 2D visualization
                    pca = PCA(n_components=2)
                    try:
                        X_pca = pca.fit_transform(X_anom)
                        fig3, ax3 = plt.subplots(figsize=(8,6))
                        palette = sns.color_palette(n_colors=len(set(labels)))
                        # plot labeled clusters
                        for lab in sorted(set(labels)):
                            mask = labels == lab
                            label_name = f"cluster {lab}" if lab != -1 else "noise"
                            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label_name, s=40, alpha=0.8)
                        ax3.set_title("PCA projection of anomaly samples (colored by DBSCAN label)")
                        ax3.legend()
                        st.pyplot(fig3)
                    except Exception:
                        st.info("PCA projection not possible for this dataset.")

                    # Representative samples per cluster
                    original_test_df = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=feature_names) if hasattr(scaler, "mean_") else pd.DataFrame(X_test_scaled, columns=feature_names)
                    # apply mask to original_test_df
                    original_anom_df = original_test_df[anomaly_mask].reset_index(drop=True)

                    reps = cluster_representatives(X_anom, labels, feature_names, original_anom_df, n_rep=3)
                    st.subheader("Cluster Representatives (closest samples to cluster centroid)")
                    if reps:
                        for lab, info in reps.items():
                            st.markdown(f"**Cluster {lab}** — size: {info['size']}")
                            st.dataframe(info['representatives'].reset_index(drop=True))
                    else:
                        st.info("No clusters (other than noise) found by DBSCAN with these parameters.")

            # -----------------------------
            # Predict on Custom Input (preserve original behaviour)
            # -----------------------------
            st.header("🔮 Predict on Custom Input")
            with st.form("predict_form"):
                user_input = {}
                for col in feature_names:
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

                    # Offer to run root cause automatically for the custom sample
                    if st.button("Run root cause on this custom sample"):
                        try:
                            full_table, top5 = root_cause_for_sample(
                                sample_scaled=input_scaled.flatten(),
                                X_train_scaled=X_train_scaled,
                                y_train=y_train.values,
                                X_train_df=pd.DataFrame(scaler.inverse_transform(X_train_scaled), columns=feature_names) if hasattr(scaler, "mean_") else pd.DataFrame(X_train_scaled, columns=feature_names),
                                feature_names=feature_names,
                                k_neighbors=20
                            )
                            st.subheader("Top feature differences (custom sample vs mean of nearest normal samples)")
                            st.table(top5[["feature", "mean_normal", "sample_value", "diff", "percent_change"]].assign(
                                percent_change=lambda df_: df_["percent_change"].map(lambda v: f"{v:.1f}%")
                            ))
                        except Exception as e:
                            st.error(f"Root cause for custom sample failed: {e}")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("📂 Upload your LTE dataset CSV to start.")
