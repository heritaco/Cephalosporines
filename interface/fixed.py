# cephalo_predictor_medication.py
import sys
import sqlite3
import csv
import json
import random
import os
import joblib
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QGroupBox, QFormLayout,
    QLineEdit, QComboBox, QPushButton, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea, QProgressBar, QHBoxLayout,
    QDialog, QCompleter
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QStringListModel


# ---------- Full SIDE_EFFECTS list ----------
SIDE_EFFECTS = [
    "Blood and lymphatic system disorders",
    "Cardiac disorders",
    "Congenital, familial and genetic disorders",
    "Ear and labyrinth disorders",
    "Endocrine disorders",
    "Eye disorders",
    "Gastrointestinal disorders",
    "General disorders and administration site conditions",
    "Hepatobiliary disorders",
    "Immune system disorders",
    "Infections and infestations",
    "Injury, poisoning and procedural complications",
    "Investigations",
    "Metabolism and nutrition disorders",
    "Musculoskeletal and connective tissue disorders",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "Nervous system disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Reproductive system and breast disorders",
    "Respiratory, thoracic and mediastinal disorders",
    "Skin and subcutaneous tissue disorders",
    "Social circumstances",
    "Surgical and medical procedures",
    "Vascular disorders"
]

CSV_PATH = os.path.join(os.getcwd(), "database.csv")  # expects database.csv in current working directory


def load_med_list_from_csv(path):
    """
    Load medication names from column C (3rd column, index 2) of a CSV file.
    Deduplicate case-insensitively, but preserve first-seen original casing.
    Returns a list of unique drug names.
    """
    meds = []
    seen = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            # skip header row
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                drug = row[2].strip()
                if not drug:
                    continue
                key = drug.lower()
                if key not in seen:
                    seen.add(key)
                    meds.append(drug)
    except Exception as e:
        print(f"Warning: could not read medication list from {path}: {e}")
    return meds


class PatientHistoryDialog(QDialog):
    def __init__(self, parent, conn):
        super().__init__(parent)
        self.conn = conn
        self.setWindowTitle("Patient History")
        self.resize(900, 500)

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "ID", "Name", "Age", "Sex", "Cephalosporin", "Weight", "Height", "Medications"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)
        self.load_data()

    def load_data(self):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, name, age, sex, cephalosporin, weight, height, medications_json
            FROM patients
            ORDER BY id DESC
        """)
        rows = cur.fetchall()
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            pid, name, age, sex, ceph, weight, height, meds_json = row
            meds_text = ""
            if meds_json:
                try:
                    meds_dict = json.loads(meds_json)
                    active_meds = [k for k, v in meds_dict.items() if v]
                    meds_text = ", ".join(active_meds)
                except Exception:
                    meds_text = meds_json

            self.table.setItem(i, 0, QTableWidgetItem(str(pid)))
            self.table.setItem(i, 1, QTableWidgetItem(str(name)))
            self.table.setItem(i, 2, QTableWidgetItem(str(age)))
            self.table.setItem(i, 3, QTableWidgetItem(str(sex)))
            self.table.setItem(i, 4, QTableWidgetItem(str(ceph)))
            self.table.setItem(i, 5, QTableWidgetItem("" if weight is None else str(weight)))
            self.table.setItem(i, 6, QTableWidgetItem("" if height is None else str(height)))
            self.table.setItem(i, 7, QTableWidgetItem(meds_text))


class CephaloPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cephalosporin Side Effect Predictor — Medications")
        self.resize(1050, 850)

        # load medication list from CSV (column C)
        self.med_list = load_med_list_from_csv(CSV_PATH)

        # DB
        self.db_path = os.path.join(os.getcwd(), "patients.db")
        self.conn = sqlite3.connect(self.db_path)
        self.ensure_table_and_columns()
        self.current_patient_id = None

        # -------- Load CatBoost SOC-specific models + feature names --------
        # We support both layouts:
        #   - running from inside "interface" folder         → "catboost.joblib"
        #   - running from repo root with "interface/..."    → "interface/catboost.joblib"
        try:
            possible_model_paths = [
                os.path.join(os.getcwd(), "interface", "catboost.joblib"),
                os.path.join(os.getcwd(), "catboost.joblib"),
            ]
            model_path = None
            for p in possible_model_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            if model_path is None:
                # fall back to first option; will raise on load and be caught
                model_path = possible_model_paths[0]

            self.models = joblib.load(model_path)

            possible_feat_paths = [
                os.path.join(os.getcwd(), "interface", "feature_names.csv"),
                os.path.join(os.getcwd(), "feature_names.csv"),
            ]
            feat_path = None
            for p in possible_feat_paths:
                if os.path.exists(p):
                    feat_path = p
                    break
            if feat_path is None:
                feat_path = possible_feat_paths[0]

            self.model_features = (
                pd.read_csv(feat_path)
                .squeeze()
                .tolist()
            )

            # SOC labels: keys of the models dict
            if isinstance(self.models, dict):
                self.model_outputs = list(self.models.keys())
            else:
                # in case joblib stores a wrapper like {"models": dict, ...}
                self.model_outputs = getattr(self.models, "keys", lambda: [])()

            print(f"✅ Loaded CatBoost models from: {model_path}")
            print(f"   Number of SOC models: {len(self.models) if isinstance(self.models, dict) else 'unknown'}")
            print(f"   Feature count: {len(self.model_features)}")
        except Exception as e:
            self.models = {}
            self.model_features = []
            self.model_outputs = []
            print("⚠️ Could not load CatBoost models / features:", e)

        # UI scaffold with a global scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.setup_ui(container)
        scroll.setWidget(container)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self.apply_modern_style()

    # ---------------- Database helpers ----------------
    def ensure_table_and_columns(self):
        """Create table if missing and ensure weight, height and medications_json columns exist."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                sex TEXT,
                cephalosporin TEXT,
                weight REAL,
                height REAL,
                medications_json TEXT
            )
        """)
        self.conn.commit()

    def insert_patient(self, name, age, sex, ceph, weight, height, medications_dict):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO patients(name, age, sex, cephalosporin, weight, height, medications_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, age, sex, ceph, weight, height, json.dumps(medications_dict)))
        self.conn.commit()
        return cur.lastrowid

    def update_patient_meds(self, patient_id, medications_dict):
        cur = self.conn.cursor()
        cur.execute("""
            UPDATE patients
            SET medications_json = ?
            WHERE id = ?
        """, (json.dumps(medications_dict), patient_id))
        self.conn.commit()

    # ---------------- UI setup ----------------
    def setup_ui(self, main_widget):
        main_layout = QVBoxLayout(main_widget)

        # Top toolbar: random example + history
        toolbar = QHBoxLayout()

        random_button = QPushButton("Random Example")
        random_button.clicked.connect(self.fill_random_example)
        toolbar.addWidget(random_button)

        history_button = QPushButton("Show History")
        history_button.clicked.connect(self.show_history)
        toolbar.addWidget(history_button)

        toolbar.addStretch(1)
        main_layout.addLayout(toolbar)

        # registration / basic info
        reg_group = QGroupBox("Patient Information")
        reg_layout = QFormLayout()
        self.name_input = QLineEdit()
        self.age_input = QLineEdit()
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Female", "Male"])
        self.cephalo_combo = QComboBox()
        self.cephalo_combo.addItems(["Cefalexin", "Cefuroxime", "Ceftriaxone", "Cefepime", "Ceftaroline"])
        # NEW: weight & height
        self.weight_input = QLineEdit()
        self.height_input = QLineEdit()
        # medication single text input with completer
        self.med_input = QLineEdit()
        self.med_input.setPlaceholderText("Type medications separated by commas — suggestions will appear as you type")
        # set completer using med_list
        # --- set up medication input + robust completer for comma-separated input ---
        self.med_input = QLineEdit()
        self.med_input.setPlaceholderText("Type medications separated by commas — suggestions will appear as you type")

        # ensure med_list is available and non-empty
        print(f"DEBUG: med_list length = {len(self.med_list)}")  # remove later if you want

        # use QStringListModel for the completer
        completer_model = QStringListModel()
        completer_model.setStringList(self.med_list)  # load list into model

        # create completer and configure
        self.completer = QCompleter()
        self.completer.setModel(completer_model)
        # case insensitivity
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        # completions will match anywhere in the string (not just at the beginning)
        self.completer.setFilterMode(Qt.MatchContains)

        # connect to med_input; use a custom handler so that it completes only after the last comma
        self.completer.activated.connect(self.insert_completion)

        # When user edits the text, update the completer prefix
        self.med_input.textEdited.connect(self.update_completer_prefix)

        # attach completer to the QLineEdit
        self.med_input.setCompleter(self.completer)

        reg_layout.addRow("Full Name:", self.name_input)
        reg_layout.addRow("Age:", self.age_input)
        reg_layout.addRow("Sex:", self.sex_combo)
        reg_layout.addRow("Cephalosporin:", self.cephalo_combo)
        reg_layout.addRow("Weight (kg):", self.weight_input)
        reg_layout.addRow("Height (cm):", self.height_input)
        reg_layout.addRow("Patient Medication:", self.med_input)
        reg_group.setLayout(reg_layout)
        main_layout.addWidget(reg_group)

        # Results table (progress bars)
        results_group = QGroupBox("Prediction Results")
        res_layout = QVBoxLayout()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Side Effect", "Probability (%)", "Severity"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setRowCount(len(SIDE_EFFECTS))
        for i, name in enumerate(SIDE_EFFECTS):
            self.results_table.setItem(i, 0, QTableWidgetItem(name))
            bar = QProgressBar()
            bar.setValue(0)
            bar.setFormat("0.00%")
            bar.setAlignment(Qt.AlignCenter)
            self.results_table.setCellWidget(i, 1, bar)
            sev_item = QTableWidgetItem("Not Severe")
            sev_item.setForeground(QColor("#22c55e"))
            self.results_table.setItem(i, 2, sev_item)
        res_layout.addWidget(self.results_table)
        results_group.setLayout(res_layout)
        main_layout.addWidget(results_group)

        # Predict button
        button_layout = QHBoxLayout()
        self.predict_button = QPushButton("Predict Side Effect Probabilities")
        self.predict_button.clicked.connect(self.predict_and_save)
        button_layout.addStretch(1)
        button_layout.addWidget(self.predict_button)
        button_layout.addStretch(1)
        main_layout.addLayout(button_layout)

    # ---------------- Completer helpers ----------------
    def update_completer_prefix(self, text):
        """
        When the user types into the line edit, update the QCompleter prefix so that it
        only considers the last comma-separated token.
        """
        parts = text.split(",")
        last_part = parts[-1].strip()
        self.completer.setCompletionPrefix(last_part)

    def insert_completion(self, completion):
        """
        When the completer suggests a medication and the user accepts it, we replace the
        last comma-separated token with the chosen completion, keeping any previous meds.
        """
        text = self.med_input.text()
        parts = text.split(",")
        # trim spaces
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            parts[-1] = completion
        else:
            parts.append(completion)
        # rebuild text with comma + space delimiter
        new_text = ", ".join(parts) + ", "
        self.med_input.setText(new_text)

    # ---------------- Random example ----------------
    def fill_random_example(self):
        """Fill inputs with a random example patient and random medications."""
        self.name_input.setText(f"Patient {random.randint(1, 100)}")
        self.age_input.setText(str(random.randint(18, 85)))
        self.sex_combo.setCurrentIndex(random.randint(0, 1))
        self.cephalo_combo.setCurrentIndex(random.randint(0, self.cephalo_combo.count() - 1))
        self.weight_input.setText(str(round(random.uniform(40, 120), 1)))
        self.height_input.setText(str(round(random.uniform(140, 200), 1)))

        # pick 1–4 random meds
        if self.med_list:
            meds = random.sample(self.med_list, min(len(self.med_list), random.randint(1, 4)))
            self.med_input.setText(", ".join(meds))

    # ---------------- Prediction model (CatBoost per SOC) ----------
    def probability_model(self, age, sex, weight, height, meds_vector):
        """
        Use the trained CatBoost models (one per SOC) to predict probabilities.
        Builds a full feature vector matching the training feature columns,
        then calls each SOC-specific model on the same feature vector.
        """
        # 0) Check that models and feature list are loaded
        if not getattr(self, "models", None):
            QMessageBox.warning(self, "Model Error", "No predictive models are loaded.")
            return {
                eff: {"prob": 0.0, "severity": "Not Severe", "color": "#e2e8f0"}
                for eff in SIDE_EFFECTS
            }

        feat_names = getattr(self, "model_features", [])
        if not feat_names:
            QMessageBox.warning(self, "Model Error", "No feature list available.")
            return {
                eff: {"prob": 0.0, "severity": "Not Severe", "color": "#e2e8f0"}
                for eff in SIDE_EFFECTS
            }

        # --- 1️⃣ Build feature dict in training feature space ---
        x_dict = {f: 0.0 for f in feat_names}

        # Demographics (names must match training columns)
        if "AGE_Y" in feat_names:
            x_dict["AGE_Y"] = age
        if "WEIGHT_KG" in feat_names:
            x_dict["WEIGHT_KG"] = weight or 0.0
        if "HEIGHT_CM" in feat_names:
            x_dict["HEIGHT_CM"] = height or 0.0
        if "GENDER_CODE" in feat_names:
            # convention used in training: 1 = male, 0 = female (adjust if needed)
            x_dict["GENDER_CODE"] = 1 if str(sex).strip().lower() == "male" else 0

        # --- 2️⃣ Medication features: map med names to columns ---
        matched_meds = []
        unmatched_meds = []

        # meds_vector is a dict {drug_name: 0/1}
        for med, val in meds_vector.items():
            if val != 1:
                continue

            med_lower = str(med).lower()
            found = False
            for f in feat_names:
                f_clean = (
                    f.lower()
                    .replace(",", "")
                    .replace("(", "")
                    .replace(")", "")
                    .strip()
                )
                m_clean = (
                    med_lower
                    .replace(",", "")
                    .replace("(", "")
                    .replace(")", "")
                    .strip()
                )
                if m_clean == f_clean or m_clean in f_clean or f_clean in m_clean:
                    x_dict[f] = 1.0
                    found = True
            if found:
                matched_meds.append(med)
            else:
                unmatched_meds.append(med)

        print(f"✅ Matched meds: {matched_meds}")
        print(f"⚠️ Unmatched meds: {unmatched_meds}")

        # Optional: include n_meds if model has it
        if "N_MEDS" in feat_names:
            x_dict["N_MEDS"] = float(sum(meds_vector.values()))

        # --- 3️⃣ Convert dict → 1×d DataFrame in correct feature order ---
        x_df = pd.DataFrame([[x_dict[f] for f in feat_names]], columns=feat_names)
        nonzero_count = sum(1 for v in x_dict.values() if v != 0)
        print(f"✅ Non-zero features count: {nonzero_count}")

        # --- 4️⃣ Predict probabilities with each SOC-specific CatBoost model ---
        probs_by_soc = {}
        if isinstance(self.models, dict):
            soc_items = self.models.items()
        else:
            # if joblib stored {"models": dict, ...}
            inner = getattr(self.models, "get", lambda key, default=None: None)("models", None)
            soc_items = inner.items() if isinstance(inner, dict) else []

        for soc_name, model in soc_items:
            try:
                # CatBoostClassifier.predict_proba → array shape (1, 2) for binary
                proba = model.predict_proba(x_df)[:, 1][0]
            except Exception as e:
                print(f"⚠️ Prediction error for SOC '{soc_name}': {e}")
                proba = 0.0
            p_pct = float(np.clip(proba * 100.0, 0.0, 100.0))
            probs_by_soc[soc_name] = p_pct

        # --- 5️⃣ Map probabilities to the fixed SIDE_EFFECTS list for the UI ---
        summary = {}
        for eff in SIDE_EFFECTS:
            p = probs_by_soc.get(eff, 0.0)  # if missing, assume 0

            if p < 33.0:
                sev = "Not Severe"
                color = "#22c55e"
            elif p < 66.0:
                sev = "Severe"
                color = "#facc15"
            else:
                sev = "Critical"
                color = "#ef4444"

            summary[eff] = {
                "prob": round(p, 2),
                "severity": sev,
                "color": color,
            }

        return summary

    # ---------------- Medication vector parsing ----------------
    def parse_med_input_to_vector(self, med_text):
        """
        Parse comma-separated medication text into a dict of {drug_name: 0/1}.
        We'll simply mark presence = 1, absence = 0.
        """
        meds_vector = {}
        if not med_text.strip():
            return meds_vector

        meds = [m.strip() for m in med_text.split(",") if m.strip()]
        for m in meds:
            meds_vector[m] = 1

        return meds_vector

    # ---------------- Predict + save to DB ----------------
    def predict_and_save(self):
        name = self.name_input.text().strip()
        age_text = self.age_input.text().strip()
        sex = self.sex_combo.currentText()
        ceph = self.cephalo_combo.currentText()
        weight_text = self.weight_input.text().strip()
        height_text = self.height_input.text().strip()
        med_text = self.med_input.text().strip()

        if not name or not age_text:
            QMessageBox.warning(self, "Missing Info", "Please enter name and age.")
            return
        try:
            age = int(age_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Age must be an integer.")
            return

        try:
            weight = float(weight_text) if weight_text else None
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Weight must be numeric.")
            return

        try:
            height = float(height_text) if height_text else None
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Height must be numeric.")
            return

        meds_vector = self.parse_med_input_to_vector(med_text)

        # Call placeholder model
        summary = self.probability_model(age, sex, weight, height, meds_vector)

        # Update UI table with bars & severity
        for i, eff in enumerate(SIDE_EFFECTS):
            d = summary[eff]
            bar = QProgressBar()
            bar.setValue(int(d["prob"]))
            bar.setFormat(f"{d['prob']}%")
            bar.setAlignment(Qt.AlignCenter)
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {d['color']}; border-radius: 5px; }}")
            self.results_table.setCellWidget(i, 1, bar)

            sev_item = QTableWidgetItem(d["severity"])
            sev_color = QColor(d["color"])
            sev_item.setForeground(sev_color)
            self.results_table.setItem(i, 2, sev_item)

        # Save to DB
        patient_id = self.insert_patient(name, age, sex, ceph, weight, height, meds_vector)
        self.current_patient_id = patient_id
        QMessageBox.information(self, "Saved", f"Patient record saved with ID {patient_id}.")

    # ---------------- History dialog ----------------
    def show_history(self):
        dlg = PatientHistoryDialog(self, self.conn)
        dlg.exec_()

    # ---------------- Styling ----------------
    def apply_modern_style(self):
        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                font-weight: bold;
                margin-top: 8px;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
            QLabel {
                font-size: 10pt;
            }
            QLineEdit, QComboBox {
                padding: 4px;
                border-radius: 4px;
                border: 1px solid #d1d5db;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QTableWidget { background-color: #ffffff; border-radius: 4px; gridline-color: #e5e7eb; selection-background-color: #bfdbfe; }
            QHeaderView::section { background-color: #e5e7eb; padding: 8px; border: none; font-weight: 600; }
            QProgressBar { border: 1px solid #d1d5db; border-radius: 5px; background-color: #f3f4f6; }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CephaloPredictor()
    w.show()
    sys.exit(app.exec_())
