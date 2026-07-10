# =========================================================
# HealthConnect - Disease Prediction from Symptoms
# Random Forest Classifier | 246,945 rows x 378 cols x 773 diseases
#
# Note on XGBoost: NOT used here. With 773 classes, XGBoost's
# multiclass objective builds trees per-class-per-round internally
# (773 x n_estimators trees), which is too heavy for free Colab
# RAM/time. RandomForest handles all classes per tree natively,
# so it scales far better for this many classes. Stick with RF.
# =========================================================

# 1. Upload / mount your full CSV in Colab first:
#    from google.colab import drive
#    drive.mount('/content/drive')
#    df = pd.read_csv('/content/drive/MyDrive/your_full_dataset.csv')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---- 1. Load data ----
df = pd.read_csv('your_full_dataset.csv')   # replace with full 246945-row file
print(df.shape)
print(df['diseases'].nunique(), "unique diseases")

# ---- 1b. Shrink memory: symptom cols are 0/1, no need for float64 ----
symptom_cols = df.columns.drop('diseases')
df[symptom_cols] = df[symptom_cols].astype('int8')   # 8x smaller than float64
print(df.memory_usage(deep=True).sum() / 1e6, "MB after downcast")

# ---- 1c. OPTIONAL: subsample if still crashing on full 246,945 rows ----
# Set USE_SAMPLE = True to train on a smaller chunk first and confirm
# the pipeline runs, then set to False later to use full data.
USE_SAMPLE = True
if USE_SAMPLE:
    df = df.sample(n=100000, random_state=42).reset_index(drop=True)
    print("Subsampled to:", df.shape)

# ---- 2. Handle rare/singleton classes ----
# train_test_split(stratify=...) crashes if any class has < 2 rows
# (can't put 1 sample into both train and test).
counts = df['diseases'].value_counts()
print((counts == 1).sum(), "diseases with only 1 row (will be dropped)")

MIN_SAMPLES = 2   # raise to 5-10 later if you want sturdier per-class learning
valid_diseases = counts[counts >= MIN_SAMPLES].index
df = df[df['diseases'].isin(valid_diseases)].reset_index(drop=True)

print("After filtering:", df.shape, "|", df['diseases'].nunique(), "diseases remain")

# ---- 3. Split features / target ----
X = df.drop(columns=['diseases'])
y = df['diseases']

le = LabelEncoder()
y_enc = le.fit_transform(y)

# ---- 4. Train/test split (safe now, no singleton classes) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ---- 5. Train Random Forest ----
import gc
del df   # free the original dataframe, X/y already hold what we need
gc.collect()

model = RandomForestClassifier(
    n_estimators=100,       # reduced from 200 - fewer trees = less RAM
    max_depth=25,           # caps tree size - prevents runaway memory on 773 classes
    min_samples_leaf=3,     # smaller/fewer nodes per tree
    n_jobs=2,               # limit parallel workers - each worker holds its own memory copy
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ---- 6. Evaluate (safe for 700+ classes) ----
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
print("\nBest predicted diseases:")
print(report_df.sort_values('f1-score', ascending=False).head(15))
print("\nWorst predicted diseases:")
print(report_df.sort_values('f1-score', ascending=True).head(15))

# ---- 7. Feature importance (for your precautions/explainability section) ----
importances = pd.Series(model.feature_importances_, index=X.columns)
top_symptoms = importances.sort_values(ascending=False).head(20)
print(top_symptoms)

# ---- 8. Save model + label encoder for Flask backend ----
joblib.dump(model, 'disease_predictor_rf.joblib')
joblib.dump(le, 'disease_label_encoder.joblib')

# ---- 9. Example: predict for a new patient symptom vector ----
def predict_disease(symptom_dict):
    """
    symptom_dict: {'sharp chest pain': 1, 'dizziness': 0, ...}
    Must include ALL 377 symptom columns (missing ones default to 0).
    """
    row = pd.Series(0, index=X.columns)
    for symptom, val in symptom_dict.items():
        if symptom in row.index:
            row[symptom] = val
    pred_enc = model.predict([row.values])[0]
    disease = le.inverse_transform([pred_enc])[0]
    return disease

# Example call - actually run it (no # in front):
result = predict_disease({'sharp chest pain': 1, 'dizziness': 1, 'palpitations': 1})
print("Predicted disease:", result)
