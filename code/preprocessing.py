# === preprocessing.py ===
# Label Encoding, Scaling, SMOTE + επιστροφή encoders για deploy/inference

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

def preprocess_data(data, target='Heart_Condition', test_size=0.2, scaling=True, handle_missing='median'):
    data = data.copy()

    # === Αφαίρεση timestamp
    if 'timestamp' in data.columns:
        data.drop(columns=['timestamp'], inplace=True)

    # === Διαχείριση Missing Values
    if handle_missing == 'median':
        data.fillna(data.median(numeric_only=True), inplace=True)
    elif handle_missing == 'mean':
        data.fillna(data.mean(numeric_only=True), inplace=True)
    else:
        raise ValueError("handle_missing πρέπει να είναι 'median' ή 'mean'.")

    # === Label Encoding για κατηγορικά features
    feature_encoders = {}
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.drop(target, errors='ignore')

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        feature_encoders[col] = le

    # === Encoding στόχου
    target_encoder = None
    if data[target].dtype == 'object' or str(data[target].dtype).startswith("category"):
        target_encoder = LabelEncoder()
        data[target] = target_encoder.fit_transform(data[target].astype(str))

    # === Ορισμός X και y
    X = data.drop(columns=[target])
    y = data[target]

    # === Scaling
    if scaling:
        scaler = StandardScaler()
        X[X.columns] = scaler.fit_transform(X)
    else:
        scaler = None

    # === SMOTE
    print("[SMOTE] Πριν:", Counter(y))
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print("[SMOTE] Μετά:", Counter(y_res))

    # === Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test, X.columns.tolist(), scaler, feature_encoders, target_encoder
