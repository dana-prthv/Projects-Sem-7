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

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="LTE Anomaly Detection with Root Cause Analysis", layout="wide")
st.title("📶 LTE Cellular Network Anomaly Detection — Root Cause Analysis & Clustering")

# -----------------------------
# Helper functions
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
    """Finds nearest normal samples, computes differences and percent changes."""
    normal_idx = np.where(y_train == 0)[0]
    if len(normal_idx) == 0:
        raise ValueError("No normal samples found in training set to compare against.")
    X_normals = X_train_scaled[normal_idx]

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X_normals)), metric="euclidean")
    nn.fit(X_normals)
    dists, idxs = nn.kneighbors(sample_scaled.reshape(1, -1), return_distance=True)
    nearest_normals = X_normals[idxs[0]]
    mean_normal = nearest_normals.mean(axis=0)
    diff = sample_scaled.flatten() - mean_normal
    abs_diff = np.abs(diff)
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
    top5 = table_sorted.head(5).copy()
    return table_sorted, top5

@st.cache_data
def anomaly_clustering(X_anomaly_scaled: np.ndarray,
                       eps: float = 0.5,
                       min_samples: int = 5) -> Tuple[np.ndarray, DBSCAN]:
    """Run DBSCAN clustering on anomaly samples."""
    if X_anomaly_scaled.shape[0] == 0:
        return np.array([]), None
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X_anomaly_scaled)
    return labels, db

def cluster_representatives(X_scaled: np.ndarray, labels: np.ndarray, feature_names: list, original_df: pd.DataFrame, n_rep: int = 3):
    """Finds representative samples (closest to cluster centroid) for each cluster."""
    reps = {}
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        if lab == -1:
            continue
        indices = np.where(labels == lab)[0]
        cluster_points = X_scaled[indices]
        centroid = cluster_points.mean(axis=0)
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
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload LTE dataset CSV", type="csv")

if uploaded_file is not None:
    try:
        df = safe_read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        df = df.drop_duplicates()

        # -----------------------------
        # Preprocessing
        # -----------------------------
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
            st.error("❌ Missing target column 'Unusual' (0 = normal, 1 = anomaly).")
            st.stop()

        X = df.drop(columns=["Unusual"])
        y = df["Unusual"].astype(int)
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            st.warning(f"Dropping non-numeric columns: {non_numeric_cols}")
            X = X.drop(columns=non_numeric_cols)
        feature_names = list(X.columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        st.success("✅ Preprocessing complete!")

        # -----------------------------
        # Sidebar — Model Selection
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

            st.success(f"✅ Model '{model_choice}' trained successfully!")

            # -----------------------------
            # Evaluation
            # -----------------------------
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
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

            # Plots
            st.subheader("Evaluation Plots")
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
            axes[0, 0].set_title("Confusion Matrix")

            if len(fpr) > 0:
                axes[0, 1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                axes[0, 1].plot([0, 1], [0, 1], 'k--')
                axes[0, 1].set_title("ROC Curve")
                axes[0, 1].legend()

            if len(rec) > 0:
                axes[1, 0].plot(rec, prec)
                axes[1, 0].set_title("Precision-Recall Curve")

            axes[1, 1].axis('off')
            try:
                if hasattr(model, "feature_importances_"):
                    fi_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
                    sns.barplot(x="importance", y="feature", data=fi_df.head(10), ax=axes[1, 1])
                    axes[1, 1].set_title("Top Feature Importances")
            except Exception:
                pass
            st.pyplot(fig)

            # -----------------------------
            # ROOT CAUSE ANALYSIS
            # -----------------------------
            st.header("🔎 Root Cause Analysis (per-sample)")
            st.markdown("""
            Compare an anomaly sample with the most similar normal samples to identify which KPIs deviate the most.
            This helps explain *why* the model flagged the sample as unusual.
            """)

            test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
            test_df["Unusual_true"] = list(y_test.values)
            test_df["Pred"] = list(y_pred)
            st.dataframe(test_df.head(10))

            sample_idx = st.number_input("Select test sample index", min_value=0, max_value=len(test_df)-1, value=0)
            analyze_button = st.button("Run Root Cause on selected sample")

            if analyze_button:
                try:
                    full_table, top5 = root_cause_for_sample(
                        sample_scaled=X_test_scaled[sample_idx],
                        X_train_scaled=X_train_scaled,
                        y_train=y_train.values,
                        X_train_df=pd.DataFrame(scaler.inverse_transform(X_train_scaled), columns=feature_names),
                        feature_names=feature_names,
                        k_neighbors=20
                    )

                    st.subheader("Top feature differences (scaled units)")
                    st.table(top5[["feature", "mean_normal", "sample_value", "diff", "percent_change"]])

                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    sns.barplot(x="diff", y="feature", data=full_table.head(10), ax=ax2)
                    ax2.set_title("Top 10 feature deviations (sample vs normal)")
                    st.pyplot(fig2)

                    st.subheader("Full feature difference table (top 20)")
                    st.dataframe(full_table.head(20))

                    # 🔍 Dynamic Explainability summary
                    st.markdown("### 🧠 **Explainable AI Summary**")
                    
                    # Analyze top deviations
                    top_increased = top5[top5['diff'] > 0].sort_values('abs_diff', ascending=False)
                    top_decreased = top5[top5['diff'] < 0].sort_values('abs_diff', ascending=False)
                    
                    summary_parts = []
                    summary_parts.append(f"**Sample #{sample_idx}** was flagged as **{'ANOMALY' if test_df.iloc[sample_idx]['Pred'] == 1 else 'NORMAL'}** by the model.")
                    
                    if len(top_increased) > 0:
                        top_inc_feature = top_increased.iloc[0]
                        summary_parts.append(f"\n**📈 Key Increase:** `{top_inc_feature['feature']}` increased by **{top_inc_feature['percent_change']:.1f}%** (from {top_inc_feature['mean_normal']:.3f} to {top_inc_feature['sample_value']:.3f})")
                        
                    if len(top_decreased) > 0:
                        top_dec_feature = top_decreased.iloc[0]
                        summary_parts.append(f"\n**📉 Key Decrease:** `{top_dec_feature['feature']}` decreased by **{abs(top_dec_feature['percent_change']):.1f}%** (from {top_dec_feature['mean_normal']:.3f} to {top_dec_feature['sample_value']:.3f})")
                    
                    # Count severity levels
                    high_deviation = len(top5[top5['abs_diff'] > top5['abs_diff'].mean()])
                    summary_parts.append(f"\n**⚠️ Severity:** {high_deviation} out of top 5 features show above-average deviation from normal behavior.")
                    
                    # List all top features
                    top_features_list = ", ".join([f"`{row['feature']}`" for _, row in top5.iterrows()])
                    summary_parts.append(f"\n**🔍 Root Cause Features:** {top_features_list}")
                    
                    summary_parts.append("\n**💡 Interpretation:** These KPI deviations represent the primary factors distinguishing this sample from normal network behavior. Large percentage changes indicate the most significant contributing factors to the anomaly.")
                    
                    st.markdown("\n".join(summary_parts))

                except Exception as e:
                    st.error(f"Root cause analysis failed: {e}")

            # -----------------------------
            # ANOMALY CLUSTERING
            # -----------------------------
            st.header("🧩 Anomaly Clustering (DBSCAN)")
            st.markdown("""
            Cluster anomaly samples to identify *distinct anomaly types* and their common feature patterns.
            """)

            cluster_mode = st.selectbox("Cluster mode", ["Predicted anomalies (model)", "Ground-truth anomalies (dataset)"])
            eps = st.slider("DBSCAN eps", 0.1, 5.0, 0.8, 0.1)
            min_samples = st.slider("DBSCAN min_samples", 2, 50, 5, 1)
            run_cluster = st.button("Run Clustering")

            if run_cluster:
                anomaly_mask = (y_pred == 1) if cluster_mode.startswith("Predicted") else (y_test.values == 1)
                X_anom = X_test_scaled[anomaly_mask]
                if X_anom.shape[0] == 0:
                    st.warning("No anomalies found to cluster.")
                else:
                    labels, db = anomaly_clustering(X_anom, eps=eps, min_samples=min_samples)
                    st.write(f"Detected {len(set(labels)) - (1 if -1 in labels else 0)} clusters, {np.sum(labels==-1)} noise samples.")
                    unique, counts = np.unique(labels, return_counts=True)
                    st.table(pd.DataFrame({"Cluster": unique, "Count": counts}))

                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_anom)
                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    for lab in sorted(set(labels)):
                        mask = labels == lab
                        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {lab}" if lab != -1 else "Noise", s=40)
                    ax3.legend()
                    ax3.set_title("PCA Projection of Anomalies (DBSCAN Clusters)")
                    st.pyplot(fig3)

                    original_test_df = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=feature_names)
                    reps = cluster_representatives(X_anom, labels, feature_names, original_test_df[anomaly_mask].reset_index(drop=True))
                    for lab, info in reps.items():
                        st.markdown(f"**Cluster {lab}** — size: {info['size']}")
                        st.dataframe(info['representatives'])

                    # 🧩 Dynamic Explainability summary for clustering
                    st.markdown("### 🧩 **Explainable AI Summary - Anomaly Clustering**")
                    
                    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    num_noise = np.sum(labels == -1)
                    total_anomalies = len(X_anom)
                    
                    summary_parts = []
                    summary_parts.append(f"**Clustering Results:** Identified **{num_clusters} distinct anomaly pattern(s)** from {total_anomalies} anomalous samples.")
                    
                    if num_noise > 0:
                        noise_pct = (num_noise / total_anomalies) * 100
                        summary_parts.append(f"\n**🔴 Outliers:** {num_noise} samples ({noise_pct:.1f}%) classified as noise - these are unique anomalies that don't fit established patterns.")
                    
                    # Analyze cluster sizes
                    cluster_sizes = {lab: np.sum(labels == lab) for lab in set(labels) if lab != -1}
                    if cluster_sizes:
                        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
                        largest_size = cluster_sizes[largest_cluster]
                        largest_pct = (largest_size / total_anomalies) * 100
                        summary_parts.append(f"\n**📊 Dominant Pattern:** Cluster {largest_cluster} is the largest with {largest_size} samples ({largest_pct:.1f}%), suggesting a common recurring network issue.")
                        
                        if len(cluster_sizes) > 1:
                            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
                            distribution = ", ".join([f"C{lab}: {size}" for lab, size in sorted_clusters[:3]])
                            summary_parts.append(f"\n**🎯 Distribution:** {distribution}")
                    
                    # Analyze cluster characteristics
                    if reps:
                        summary_parts.append(f"\n**🔬 Pattern Analysis:**")
                        for lab, info in list(reps.items())[:3]:  # Top 3 clusters
                            cluster_data = X_anom[labels == lab]
                            centroid_features = info['centroid']
                            # Find top 3 features with highest absolute values in centroid
                            feature_importance = [(feature_names[i], abs(centroid_features[i])) for i in range(len(feature_names))]
                            top_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:3]
                            feature_list = ", ".join([f"`{feat}`" for feat, _ in top_features])
                            summary_parts.append(f"\n  - **Cluster {lab}** ({info['size']} samples): Characterized by deviations in {feature_list}")
                    
                    summary_parts.append("\n\n**💡 Actionable Insights:** Each cluster represents a distinct network failure mode. Focus remediation efforts on the largest clusters first, while investigating noise samples for emerging issues.")
                    
                    st.markdown("\n".join(summary_parts))

            # -----------------------------
            # CUSTOM PREDICTION
            # -----------------------------
            st.header("🔮 Predict on Custom Input")
            with st.form("predict_form"):
                user_input = {}
                for col in feature_names:
                    min_val, max_val, mean_val = float(df[col].min()), float(df[col].max()), float(df[col].mean())
                    user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val, step=0.01)
                submitted = st.form_submit_button("Predict")

            if submitted:
                input_df = pd.DataFrame([user_input])
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

                if pred == 1:
                    st.error("🚨 Prediction: **Unusual (Anomaly detected)**")
                else:
                    st.success("✅ Prediction: **Usual (Normal behavior)**")

                if prob is not None:
                    st.info(f"Prediction Confidence: {prob:.3f}")

                if st.button("Run root cause on this custom sample"):
                    try:
                        full_table, top5 = root_cause_for_sample(
                            sample_scaled=input_scaled.flatten(),
                            X_train_scaled=X_train_scaled,
                            y_train=y_train.values,
                            X_train_df=pd.DataFrame(scaler.inverse_transform(X_train_scaled), columns=feature_names),
                            feature_names=feature_names,
                            k_neighbors=20
                        )
                        st.subheader("Top feature differences (custom sample vs nearest normal)")
                        st.table(top5[["feature", "mean_normal", "sample_value", "diff", "percent_change"]].assign(
                            percent_change=lambda df_: df_["percent_change"].map(lambda v: f"{v:.1f}%")
                        ))

                        # Dynamic Explainability summary for custom input
                        st.markdown("### 🧠 **Explainable AI Summary - Custom Input**")
                        
                        top_increased_custom = top5[top5['diff'] > 0].sort_values('abs_diff', ascending=False)
                        top_decreased_custom = top5[top5['diff'] < 0].sort_values('abs_diff', ascending=False)
                        
                        summary_parts_custom = []
                        summary_parts_custom.append(f"**Custom Input Classification:** {'🚨 ANOMALY DETECTED' if pred == 1 else '✅ NORMAL BEHAVIOR'}")
                        
                        if prob is not None:
                            confidence_level = "high" if prob > 0.8 or prob < 0.2 else "moderate" if prob > 0.6 or prob < 0.4 else "low"
                            summary_parts_custom.append(f" (Confidence: {prob:.1%} - {confidence_level})")
                        
                        if len(top_increased_custom) > 0:
                            top_inc = top_increased_custom.iloc[0]
                            summary_parts_custom.append(f"\n\n**📈 Highest Increase:** `{top_inc['feature']}` is **{top_inc['percent_change']:.1f}%** above normal (expected: {top_inc['mean_normal']:.3f}, actual: {top_inc['sample_value']:.3f})")
                            
                        if len(top_decreased_custom) > 0:
                            top_dec = top_decreased_custom.iloc[0]
                            summary_parts_custom.append(f"\n**📉 Highest Decrease:** `{top_dec['feature']}` is **{abs(top_dec['percent_change']):.1f}%** below normal (expected: {top_dec['mean_normal']:.3f}, actual: {top_dec['sample_value']:.3f})")
                        
                        # Overall deviation assessment
                        avg_abs_change = top5['abs_diff'].mean()
                        max_abs_change = top5['abs_diff'].max()
                        deviation_level = "severe" if max_abs_change > 2 * avg_abs_change else "moderate" if max_abs_change > avg_abs_change else "minor"
                        
                        summary_parts_custom.append(f"\n\n**⚠️ Deviation Level:** {deviation_level.capitalize()} - The largest deviation is {max_abs_change:.3f} standard deviations from normal.")
                        
                        # Recommendation
                        if pred == 1:
                            top_features_str = ", ".join([f"`{row['feature']}`" for _, row in top5.head(3).iterrows()])
                            summary_parts_custom.append(f"\n\n**🔧 Recommended Action:** Investigate {top_features_str} as primary contributors to the anomaly. These KPIs show the most significant deviations from typical network behavior.")
                        else:
                            summary_parts_custom.append(f"\n\n**✅ Status:** All KPIs within acceptable ranges. No immediate action required.")
                        
                        st.markdown("\n".join(summary_parts_custom))

                    except Exception as e:
                        st.error(f"Root cause for custom sample failed: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("📂 Upload your LTE dataset CSV to start.")