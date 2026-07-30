import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import IsolationForest
from preprocess import preprocessing
from feat import train_features
from anomalies import inject_anomalies
from sklearn.metrics import (confusion_matrix,classification_report,precision_score,recall_score,f1_score,accuracy_score)

df = pd.read_csv("grid-power-quality-ml/src/household_power_consumption.txt", sep=";", na_values="?")
df = preprocessing(df)
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_raw = df.iloc[:train_end].copy()
validation_raw = df.iloc[train_end:val_end].copy()
test_raw = df.iloc[val_end:].copy()


test_eval_raw = inject_anomalies(test_raw, num_events=500)

test_eval = train_features(test_eval_raw)

train = train_features(train_raw)
validation = train_features(validation_raw)
test_eval = train_features(test_eval_raw)

feature_columns = ["Voltage", "Global_intensity", "Frequency", "apparent_power", 
                     "power_factor", "voltage_deviation", "frequency_deviation", "voltage_mean", 
                     "frequency_mean", "current_mean", "active_power_mean", "reactive_power_mean", "voltage_std",
                     "frequency_std", "current_std", "active_power_std", "reactive_power_std" ,
                     "voltage_change" , "frequency_change", "current_change", "active_power_change",
                     "reactive_power_change", "weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                       "month_sin", "month_cos"]


X_train = train[feature_columns]

X_val = validation[feature_columns]

X_test = test_eval[feature_columns]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)
#contamination_rates = [0.005, 0.0001, 0.00005]
model = IsolationForest( n_estimators=200,contamination=0.005, random_state=42, n_jobs=-1)
model.fit(X_train)

val_scores = model.decision_function(X_val)

#print(f"Min score (worst normal point): {val_scores.min():.4f}")
#print(f"0.01% percentile:               {np.percentile(val_scores, 0.01):.4f}")
#print(f"0.1% percentile:                {np.percentile(val_scores, 0.1):.4f}")

custom_threshold = 0.0936
#np.percentile(val_scores, 3)
val_pred_custom = np.where(val_scores < custom_threshold, 1, 0)
false_positives = np.sum(val_pred_custom)
#print(false_positives)

print(f"\nFalse Positives with Custom Threshold: {false_positives} / {len(X_val)}")

scores = model.decision_function(X_test)
pred =  (scores < custom_threshold).astype(int)

results = test_eval.copy()

results["anomaly_score"] = scores
results["prediction"] = pred

results.to_csv("predictions.csv", index=False)
cm = confusion_matrix(test_eval["label"], pred)
print(cm)

print(len(train))
print(test_eval["label"].value_counts())
print(np.unique(pred, return_counts=True))

print(accuracy_score(
    test_eval["label"],
    pred
))

print(precision_score(
    test_eval["label"],
    pred,zero_division = 0
))

print(recall_score(
    test_eval["label"],
    pred
))

print(f1_score(
    test_eval["label"],
    pred
))
     