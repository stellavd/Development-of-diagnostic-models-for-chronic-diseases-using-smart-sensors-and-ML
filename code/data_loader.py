# === data_loader.py ===
# φόρτωση dataset με GUI ή direct path και logging σε αρχείο

import pandas as pd
import os
import logging

# === Logging Configuration ===
def configure_logging(log_to_file=True, log_file='logs/data_loader.log', level=logging.INFO):
    logger = logging.getLogger('data_loader')
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(message)s')

    # Console Handler (warnings and errors only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (info and above)
    if log_to_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# === Dataset Loader ===
def load_dataset(file_path=None, target='Heart_Condition', encoding='utf-8', sep=',', gui_fallback=True, logger=None):
    """
    Φορτώνει αρχείο CSV και επιστρέφει dataframe + στατιστικά.
    """
    if logger is None:
        logger = configure_logging()

    if file_path is None and gui_fallback:
        try:
            from tkinter import Tk, filedialog
            Tk().withdraw()
            file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        except Exception as e:
            logger.error("Αποτυχία χρήσης GUI: %s", e)
            raise RuntimeError("Αποτυχία χρήσης tkinter. Δώστε απευθείας file_path.")

    if not file_path or not os.path.isfile(file_path):
        logger.error("Μη έγκυρο path: %s", file_path)
        raise FileNotFoundError("Δεν δόθηκε έγκυρο path αρχείου.")

    try:
        # Ανίχνευση "κρυφών" missing values κατά την ανάγνωση
        data = pd.read_csv(file_path, encoding=encoding, sep=sep, na_values=["?", "na", "none", "None", "NaN", ""])
        logger.info("Φόρτωση αρχείου: %s", file_path)
    except Exception as e:
        logger.exception("Σφάλμα κατά τη φόρτωση CSV:")
        raise ValueError(f"Σφάλμα κατά τη φόρτωση αρχείου: {e}")

    if target not in data.columns:
        logger.error("Απουσιάζει η στήλη στόχου '%s'", target)
        raise ValueError(f"Η στήλη στόχου '{target}' δεν υπάρχει στο dataset.")

    # Στατιστικά
    stats = {
        "file_path": file_path,
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "missing": data.isnull().sum().to_dict(),
        "describe": data.describe(include='all').to_dict()
    }

    logger.info("Σχήμα: %s", data.shape)
    logger.info("Στήλες: %s", data.columns.tolist())
    logger.info("Missing Values: %d", data.isnull().sum().sum())
    logger.info("Κατανομή στόχου: %s", data[target].value_counts().to_dict())

    return data, stats
