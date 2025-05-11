# === main.py ===
# Κεντρική εκτέλεση του thesisbeta pipeline

from data_loader import load_dataset, configure_logging as configure_data_logging
from eda import run_eda
from preprocessing import preprocess_data
from modeling import run_modeling
from encoder_utils import save_encoders

def main():
    # === Logging ενεργοποίηση
    data_logger = configure_data_logging()

    # === Ορισμός κλάσεων & στόχου
    label_names = ['Arrhythmia', 'Diabetes', 'Healthy', 'Hypertension', 'Hypotension']
    target_col = 'Heart_Condition'

    # === [1] Φόρτωση δεδομένων
    data, stats = load_dataset(file_path='data/heart.csv', target=target_col, logger=data_logger)

    # === [2] Εξερεύνηση Δεδομένων
    mi_scores = run_eda(data, target=target_col)

    # === [3] Προεπεξεργασία Δεδομένων
    X_train, X_test, y_train, y_test, feature_names, scaler, feature_encoders, target_encoder = preprocess_data(
        data,
        target=target_col,
        test_size=0.2,
        handle_missing='median',
        scaling=True
    )

    # === [4] Αποθήκευση Encoders για inference
    save_encoders(feature_encoders, target_encoder)

    # === [5] Εκπαίδευση Μοντέλου και Ερμηνεία
    model, y_pred, metrics = run_modeling(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        label_names=label_names,
        cv_folds=5,
        shap_max=15,
        save_model=True
    )

    # === [6] Τελική Αναφορά
    print("\n📊 [Τελικές Μετρικές]")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC:  {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
