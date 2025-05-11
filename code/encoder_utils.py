# === encoder_utils.py ===
# Αποθήκευση και φόρτωση των encoders (target + features)

import joblib
import os

def save_encoders(feature_encoders, target_encoder, path='models/encoders'):
    os.makedirs(path, exist_ok=True)
    joblib.dump(feature_encoders, os.path.join(path, 'feature_encoders.pkl'))
    joblib.dump(target_encoder, os.path.join(path, 'target_encoder.pkl'))
    print(f"[Encoders] Αποθηκεύτηκαν στο {path}")

def load_encoders(path='models/encoders'):
    feature_encoders = joblib.load(os.path.join(path, 'feature_encoders.pkl'))
    target_encoder = joblib.load(os.path.join(path, 'target_encoder.pkl'))
    print(f"[Encoders] Φορτώθηκαν από {path}")
    return feature_encoders, target_encoder
