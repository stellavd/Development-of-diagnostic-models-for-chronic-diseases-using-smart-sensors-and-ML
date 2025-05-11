# === modeling.py ===
# Εκπαίδευση Stacking Classifier + SHAP με logging και αποθήκευση αποτελεσμάτων

import os
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)

# === Logging Configuration ===
def configure_logging(log_file='logs/modeling.log', level=logging.INFO):
    logger = logging.getLogger('modeling')
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Show all important info in console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# === Κύρια Συνάρτηση Εκπαίδευσης ===
def run_modeling(X_train, X_test, y_train, y_test, feature_names, label_names, cv_folds=5, shap_max=15, save_model=True):
    logger = configure_logging()

    logger.info("[1] Ορισμός βασικών μοντέλων...")
    base_models = [
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', tree_method='gpu_hist')),
        ('lgbm', LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0)),
        ('rf', RandomForestClassifier(n_jobs=-1))
    ]
    meta_model = LogisticRegression(max_iter=1000, n_jobs=-1)

    model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=cv_folds)
    model.fit(X_train, y_train)
    logger.info("Μοντέλο εκπαιδεύτηκε επιτυχώς.")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # === Accuracy & ROC AUC
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"ROC AUC (OvR): {roc:.4f}")

    # === Classification Report
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    os.makedirs("results", exist_ok=True)
    df_report.to_csv("results/classification_report.csv")
    logger.info("Αποθηκεύτηκε το classification_report.csv")

    # === Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    cm_display.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv("results/confusion_matrix.csv")
    logger.info("Αποθηκεύτηκε confusion_matrix σε PNG/CSV")

    # === Αξιολόγηση επιμέρους μοντέλων και αποθήκευση συγκρίσεων
    model_scores = []
    for name, clf in model.named_estimators_.items():
        yp = clf.predict(X_test)
        score = accuracy_score(y_test, yp)
        logger.info(f"[{name}] Accuracy: {score:.4f}")
        model_scores.append({'Model': name, 'Accuracy': score})

    # === Save base model comparison
    model_scores_df = pd.DataFrame(model_scores)
    model_scores_df.to_csv("results/base_model_accuracies.csv", index=False)
    logger.info("Αποθηκεύτηκε base_model_accuracies.csv")

    # === SHAP για XGBoost
    logger.info("Υπολογισμός SHAP για XGBoost...")
    explainer = shap.Explainer(model.named_estimators_['xgb'], X_train, feature_names=feature_names)
    shap_values = explainer(X_test)

    os.makedirs("results/shap", exist_ok=True)
    summary_data = []

    for i, name in enumerate(label_names):
        logger.info(f"[SHAP] Κλάση: {name}")

        # Beeswarm Plot
        plt.figure()
        shap.plots.beeswarm(shap_values[:, :, i], max_display=shap_max, show=False)
        plt.title(f'SHAP Beeswarm - Κλάση: {name}')
        plt.tight_layout()
        beeswarm_path = f"results/shap/shap_beeswarm_{name}.png"
        plt.savefig(beeswarm_path)
        plt.close()
        logger.info(f"Αποθηκεύτηκε: {beeswarm_path}")

        # Bar Plot
        plt.figure()
        shap.plots.bar(shap_values[:, :, i], max_display=shap_max, show=False)
        plt.title(f'SHAP Feature Importance - Κλάση: {name}')
        plt.tight_layout()
        bar_path = f"results/shap/shap_bar_{name}.png"
        plt.savefig(bar_path)
        plt.close()
        logger.info(f"Αποθηκεύτηκε: {bar_path}")

        # SHAP values summary table
        class_shap_values = shap_values[:, :, i].values
        mean_abs_shap = np.abs(class_shap_values).mean(axis=0)
        feature_ranking = pd.DataFrame({
            'Feature': feature_names,
            'MeanAbsSHAP': mean_abs_shap,
            'Class': name
        }).sort_values(by='MeanAbsSHAP', ascending=False).head(shap_max)
        summary_data.append(feature_ranking)

    shap_summary_df = pd.concat(summary_data, axis=0)
    shap_summary_df.to_csv("results/shap/shap_top_features_summary.csv", index=False)
    logger.info("Αποθηκεύτηκε shap_top_features_summary.csv")

    # === Αποθήκευση Μοντέλου
    if save_model:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/final_model.pkl")
        logger.info("Το εκπαιδευμένο μοντέλο αποθηκεύτηκε στο models/final_model.pkl")

    return model, y_pred, {"accuracy": acc, "roc_auc": roc}
