# === eda.py ===
# Καθαρό και modular Exploratory Data Analysis για thesisbeta pipeline

import os
import matplotlib.pyplot as plt
import seaborn as sns

# === Παλέτα για συνεπή χρωματισμό κλάσεων
palette = {
    "Healthy": "#1f77b4",
    "Hypertension": "#d62728",
    "Hypotension": "#ff7f0e",
    "Arrhythmia": "#2ca02c",
    "Diabetes": "#9467bd"
}

# === Κύρια Συνάρτηση EDA ===
def run_eda(data, target="Heart_Condition", save_dir="results"):
    print("\n[EDA] Εκκίνηση καθαρής ανάλυσης δεδομένων...")

    X = data.drop(columns=[target])
    y = data[target]
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # === Δημιουργία φακέλων
    folders = ["boxplots", "counts", "distributions"]
    for folder in folders:
        os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

    # === Συνάρτηση Αποθήκευσης
    def plot_and_save(path, show=False):
        if show:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()

    # === Επιλεγμένα χρήσιμα χαρακτηριστικά
    important_numeric = [
        "Age", "BMI", "Blood_Glucose_mg_dL",
        "Blood_Pressure_Systolic_mmHg", "Blood_Pressure_Diastolic_mmHg",
        "Heart_Rate_bpm", "Heart_Rate_Variability_ms", "Stress_Level",
        "Steps_Taken", "Calories_Burned_kcal", "ECG_Lead1_mV", "ECG_Lead2_mV"
    ]

    important_categorical = [
        "Activity_Type", "Alcohol_Consumption", "BMI_Category",
        "Diet_Type", "Gender", "Medication_Taken", "Smoking_Status"
    ]

    # === Boxplots ===
    for col in important_numeric:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=target, y=col, data=data, palette=palette, showfliers=False)
        plt.title(f"Boxplot του {col} ανά {target}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plot_and_save(f"{save_dir}/boxplots/boxplot_{col}.png")

    # === Countplots ===
    for col in important_categorical:
        plt.figure(figsize=(12, 4))
        sns.countplot(x=col, hue=target, data=data, palette=palette)
        plt.title(f"Κατηγορικό χαρακτηριστικό: {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_and_save(f"{save_dir}/counts/category_{col}.png")

    # === Distributions ===
    for col in important_numeric:
        plt.figure(figsize=(10, 4))
        sns.histplot(data=data, x=col, hue=target, kde=True, multiple="stack", palette=palette)
        plt.title(f"Κατανομή του {col} ανά {target}")
        plt.tight_layout()
        plot_and_save(f"{save_dir}/distributions/distribution_{col}.png")

    print("[EDA] Ολοκλήρωση. Τα διαγράμματα αποθηκεύτηκαν στο:", save_dir)
    return None
