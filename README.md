# grid-power-quality-ml
Machine Learning-based Grid Voltage/Frequency Anomaly Detection and Power Quality Monitoring using Isolation Forest
# Grid Voltage/Frequency Anomaly Detection & Power Quality Monitoring using Machine Learning

## Overview
This project is an ML framework to detect voltage and frequency anomalies and power quality anomalies in electrical power systems. The goal is to detect abnormal operating conditions automatically by an unsupervised learning algorithm without having to use manually labelled training data to detect power quality disturbances.
The main data source for the project is the UCI Individual Household Electric Power Consumption Dataset. Realistic synthetic power quality disturbances are added to the original data set to represent realistic grid faults and abnormal operating conditions. A model called “Isolation Forest” is then trained on normal data, and tested on data with injected anomalies.
The project also involves extensive model evaluation, threshold optimization, visualization and an interactive Streamlit dashboard for investigating the results of the anomaly detection.

# Objectives
The objectives of this project are to:
* Develop an end-to-end machine learning pipeline for power quality anomaly detection.
* Detect abnormal voltage and frequency behaviour automatically.
* Simulate realistic electrical disturbances through synthetic anomaly injection.
* Evaluate anomaly detection performance using multiple statistical metrics.
* Visualize anomaly detection results through interactive dashboards.
* Demonstrate the application of unsupervised machine learning to smart grid monitoring.

# Dataset

Dataset

UCI Individual Household Electric Power Consumption Dataset

**Sampling Rate**

* One-minute interval measurements*

*Main Measurements*

* Voltage
* Global Active Power
* Global Reactive Power
* Global Intensity
* Sub-metering values
* Timestamp

Since the dataset contains normal operating conditions only, synthetic anomalies are introduced to create realistic evaluation scenarios.

---

# Synthetic Anomaly Generation

To simulate real-world power quality disturbances, multiple anomaly types are injected into the validation and testing datasets.
Implemented anomaly types include:
* Voltage Sag
* Voltage Swell
* Frequency Drift
* Current Surge
* Load Spike

Each anomaly modifies one or more electrical variables using smooth waveform-based transitions rather than abrupt step changes, producing more realistic fault behavior.

# Feature Engineering
The following engineered features are extracted before model training.
## Electrical Features
* Apparent Power
* Power Factor
* Voltage Deviation
* Frequency Deviation

## Rolling Statistics
* Voltage Mean
* Frequency Mean
* Current Mean
* Active Power Mean
* Reactive Power Mean
* Voltage Standard Deviation
* Frequency Standard Deviation
* Current Standard Deviation
* Active Power Standard Deviation
* Reactive Power Standard Deviation

## Dynamic Features
* Voltage Rate of Change
* Frequency Rate of Change
* Current Rate of Change
* Active Power Change
* Reactive Power Change

## Temporal Features
* Weekend Indicator
* Hour (sin)
* Hour (cos)
* Day of Week (sin)
* Day of Week (cos)
* Month (sin)
* Month (cos)

All numerical features are standardized prior to model training.

---

# Machine Learning Model

The anomaly detection model is based on the Isolation Forest algorithm.
Isolation Forest is particularly suitable because:

* It is an unsupervised learning algorithm.
* It requires only normal operating data for training.
* It efficiently isolates abnormal observations.
* It scales well to large datasets.

Training consists of:
1. Training on normal operating data.
2. Computing anomaly scores using the Isolation Forest decision function.
3. Applying threshold optimization.
4. Predicting anomalous operating conditions.

---
# Model Evaluation
Model performance is evaluated using:
## Classification Metrics
* Accuracy
* Precision
* Recall
* F1 Score

## Visualization
* Confusion Matrix
* Precision–Recall Curve
* Anomaly Score Distribution
* Precision vs Threshold
* Recall vs Threshold
* F1 Score vs Threshold
* Combined Threshold Analysis

Threshold optimization is performed to identify the operating point that provides the best balance between precision and recall.

---

# Results
Current evaluation demonstrates that the Isolation Forest successfully distinguishes between normal operating conditions and injected power quality disturbances.
Current best evaluation metrics:

* Accuracy: 95.66%
* Precision: 69.22%
* Recall: 55.18%
* F1 Score: 61.41%

The model demonstrates strong discrimination between normal and anomalous observations while maintaining a low false positive rate.

> **Note:** These metrics will continue to improve as threshold optimization and anomaly generation are further refined.

---

# Project Structure
grid-power-quality-ml/

 data/
 models/
 notebooks/
│   ─ 01_EDA.ipynb
│   - 02_Model_Evaluation.ipynb
│
 results/
│   - confusion_matrix.png
│   - precision_recall_curve.png
│   - threshold_analysis.png
│   - score_distribution.png
│
 src/
│   - preprocess.py
│   - feat.py
│   - anomalies.py
│   - train.py
│   - evaluate.py
│
 app.py
 requirements.txt
 README.md




# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/grid-power-quality-ml.git
cd grid-power-quality-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Train the Model

```bash
python src/train.py
```

## Evaluate the Model

```bash
python src/evaluate.py
```

# Future Improvements

Future enhancements include:
* Real-time streaming anomaly detection
* PMU and SCADA data integration
* Autoencoder-based anomaly detection
* LSTM and Transformer-based sequence models
* Explainable AI (SHAP/LIME)
* Online learning for adaptive thresholding
* Edge deployment for embedded monitoring systems
* Cloud deployment for large-scale smart grid monitoring
  
# Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Plotly
* Streamlit

---
# License

This project is intended for educational and research purposes.

---

# Author

**Kamsi Wogu**

Bachelor of Engineering (Electrical & Electronics Engineering)

Machine Learning | Smart Grids | Power Systems | Artificial Intelligence | Autonomous Systems
