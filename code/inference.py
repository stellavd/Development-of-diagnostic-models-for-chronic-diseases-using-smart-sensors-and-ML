# === inference.py ===
# Demo πρόβλεψης με αποθηκευμένο Stacking μοντέλο + encoders

import joblib
import pandas as pd
from encoder_utils import load_encoders

# === Φόρτωση Μοντέλου + Encoders
model = joblib.load("models/final_model.pkl")
feature_encoders, target_encoder = load_encoders()

# === Παράδειγμα νέου input
raw_input = {
    'Age': 58,
    'Sex': 'Female',
    'BMI': 29.4,
    'Smoking': 'No',
    'PhysicalActivity': 'Medium',
    'BloodPressure': 130,
    'Glucose': 180
    # Πρόσθεσε όλα τα features που χρησιμοποιεί το εκπαιδευμένο μοντέλο
}

# === Προεπεξεργασία input
input_df = pd.DataFrame([raw_input])

# Εφαρμογή encoders στα κατηγορικά
for col, encoder in feature_encoders.items():
    input_df[col] = encoder.transform(input_df[col].astype(str))

# Scaling (αν χρησιμοποιείς τον ίδιο scaler, πρέπει να τον φορτώσεις με joblib)
# scaler = joblib.load('models/scaler.pkl')  # Optional
# input_df[input_df.columns] = scaler.transform(input_df)

# === Πρόβλεψη
prediction = model.predict(input_df)
predicted_label = target_encoder.inverse_transform(prediction)

print(f"\n✅ Προβλεπόμενη Διάγνωση: {predicted_label[0]}")
