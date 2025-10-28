Cellular networks are the backbone of modern mobile communication, providing seamless connectivity to millions of users. 
With increasing user demand and dynamic traffic patterns, efficient resource utilization and energy management have become critical for network operators. 
Anomalies in cellular networks, such as sudden spikes in traffic or abnormal user behavior, can degrade service quality and necessitate rapid reconfiguration of network resources. 
Detecting such anomalies in real time is therefore crucial for maintaining optimal network performance.
The primary objective of this work is to develop a machine learning-based system capable of identifying unusual behavior in LTE cellular network cells. 
Specifically, the system classifies each 15-minute sample from network cells as normal (0) or unusual (1), enabling proactive reconfiguration of base stations to handle unexpected traffic variations. 
The focus is on leveraging historical KPI traces, including PRB utilization, throughput, and active user counts, to build predictive models that detect deviations from typical behavior patterns.
The proposed methodology involves a three-layer architecture comprising a presentation layer (Streamlit UI for data upload, visualization, and custom predictions), an application layer (data preprocessing, model training, evaluation, root cause analysis, and clustering), and a data layer (historical LTE KPI dataset). 
Data preprocessing includes handling categorical and numeric features, scaling, and train-test splitting. Multiple machine learning models are trained and evaluated, including Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Logistic Regression, and MLP.
Results indicate that tree-based ensemble models outperform others, with XGBoost providing strong accuracy, CatBoost performing better than most, and LightGBM achieving the highest accuracy above 99.5%. Random Forest achieves 80–90% accuracy, while MLP achieves around 85%. 
These results demonstrate the effectiveness of the proposed framework for anomaly detection, offering both high predictive performance and explainable insights into network behavior.


Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
Random Forest	0.920	0.965	0.737	0.836	0.977
Gradient Boosting	0.914	0.996	0.693	0.817	0.971
XGBoost	0.983	0.994	0.943	0.968	0.998
LightGBM	0.990	0.997	0.965	0.981	0.999
CatBoost	0.991	0.998	0.969	0.983	1.000
Logistic Regression	0.720	0.338	0.013	0.026	0.696
MLP Classifier	0.820	0.744	0.532	0.620	0.858
