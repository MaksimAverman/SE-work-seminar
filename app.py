import customtkinter as ctk
from tkinter import messagebox, filedialog
import copy
import sys
import os
import hashlib
import pandas as pd
import threading
import time
import sqlite3
import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "icu_system.db"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from app import compute_features, generate_alerts, get_risk_drivers, model, scaler, FEATURES, THRESHOLD
import numpy as np

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


USERS = {
    "doctor1": {
        "password": hash_password("doctor123"),
        "role": "Doctor",
        "name": "Dr. Sarah Cohen"
    },
    "nurse1": {
        "password": hash_password("nurse123"),
        "role": "Nurse",
        "name": "Nurse Daniel Levi"
    }
}


def init_database():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            subject_id TEXT,
            hadm_id TEXT,
            icustay_id TEXT PRIMARY KEY,
            intime TEXT,
            gender TEXT,
            age REAL,
            heart_rate_mean REAL,
            heart_rate_min REAL,
            heart_rate_max REAL,
            systolic_bp_mean REAL,
            systolic_bp_min REAL,
            systolic_bp_max REAL,
            diastolic_bp_mean REAL,
            diastolic_bp_min REAL,
            diastolic_bp_max REAL,
            creatinine_max REAL,
            lactate_max REAL,
            admit_hour INTEGER,
            admit_dayofweek INTEGER,
            icu_hours TEXT,
            admission_type TEXT,
            diagnosis TEXT,
            alert_status TEXT,
            clinical_note TEXT,
            hospital_expire_flag INTEGER,
            risk_score REAL,
            risk_level TEXT,
            room_number TEXT
        )
    """)

    cursor.execute("PRAGMA table_info(patients)")
    columns = [column[1] for column in cursor.fetchall()]
    if "room_number" not in columns:
        cursor.execute("ALTER TABLE patients ADD COLUMN room_number TEXT")

    connection.commit()
    connection.close()


def save_patient_to_database(patient):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO patients (
            subject_id, hadm_id, icustay_id, intime, gender, age,
            heart_rate_mean, heart_rate_min, heart_rate_max,
            systolic_bp_mean, systolic_bp_min, systolic_bp_max,
            diastolic_bp_mean, diastolic_bp_min, diastolic_bp_max,
            creatinine_max, lactate_max, admit_hour, admit_dayofweek,
            icu_hours, admission_type, diagnosis, alert_status,
            clinical_note, hospital_expire_flag, risk_score, risk_level, room_number
        )
        VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?, ?
        )
    """, (
        str(patient.get("subject_id", patient.get("id", ""))),
        str(patient.get("hadm_id", "")),
        str(patient.get("icustay_id", "")),
        patient.get("intime", ""),
        patient.get("gender"),
        patient.get("age"),
        patient.get("heart_rate_mean"),
        patient.get("heart_rate_min"),
        patient.get("heart_rate_max"),
        patient.get("systolic_bp_mean"),
        patient.get("systolic_bp_min"),
        patient.get("systolic_bp_max"),
        patient.get("diastolic_bp_mean"),
        patient.get("diastolic_bp_min"),
        patient.get("diastolic_bp_max"),
        patient.get("creatinine_max"),
        patient.get("lactate_max"),
        patient.get("admit_hour", 12),
        patient.get("admit_dayofweek", 2),
        str(patient.get("icu_hours", "N/A")),
        patient.get("admission_type", ""),
        patient.get("diagnosis", ""),
        patient.get("alert_status", "New"),
        patient.get("clinical_note", ""),
        patient.get("hospital_expire_flag"),
        patient.get("_risk_score"),
        patient.get("_risk_level"),
        patient.get("room_number")
    ))

    connection.commit()
    connection.close()


def load_patients_from_database():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM patients")
    rows = cursor.fetchall()
    connection.close()

    loaded_patients = []
    for idx, row in enumerate(rows, start=1):
        existing_room = row["room_number"] if "room_number" in row.keys() and row["room_number"] else f"ICU-{(idx % 15) + 101}"

        patient = {
            "id": row["subject_id"],
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "icustay_id": row["icustay_id"],
            "intime": row["intime"],
            "gender": row["gender"],
            "age": row["age"],
            "heart_rate_mean": row["heart_rate_mean"],
            "heart_rate_min": row["heart_rate_min"],
            "heart_rate_max": row["heart_rate_max"],
            "systolic_bp_mean": row["systolic_bp_mean"],
            "systolic_bp_min": row["systolic_bp_min"],
            "systolic_bp_max": row["systolic_bp_max"],
            "diastolic_bp_mean": row["diastolic_bp_mean"],
            "diastolic_bp_min": row["diastolic_bp_min"],
            "diastolic_bp_max": row["diastolic_bp_max"],
            "creatinine_max": row["creatinine_max"],
            "lactate_max": row["lactate_max"],
            "admit_hour": row["admit_hour"],
            "admit_dayofweek": row["admit_dayofweek"],
            "icu_hours": row["icu_hours"],
            "admission_type": row["admission_type"],
            "diagnosis": row["diagnosis"],
            "alert_status": row["alert_status"],
            "clinical_note": row["clinical_note"],
            "hospital_expire_flag": row["hospital_expire_flag"],
            "room_number": existing_room
        }

        if row["risk_score"] is not None:
            patient["_risk_score"] = float(row["risk_score"])
        if row["risk_level"]:
            patient["_risk_level"] = row["risk_level"]

        loaded_patients.append(patient)

    return loaded_patients


init_database()
patients = load_patients_from_database()


def get_reassessment_info(patient):
    level = patient.get("_risk_level", "LOW")
    last_review = patient.get("last_reviewed_at")

    if not last_review:
        if level in ["CRITICAL", "HIGH"]:
            return "⚠️ Urgent: Pending initial review!", "#c0392b"
        return "ℹ️ Pending review", "#e67e22"

    review_intervals = {
        "CRITICAL": 15,
        "HIGH": 30,
        "MODERATE": 60,
        "LOW": 240
    }

    interval = review_intervals.get(level, 60)

    try:
        last_dt = datetime.datetime.strptime(last_review, "%Y-%m-%d %H:%M")
        elapsed_minutes = (datetime.datetime.now() - last_dt).total_seconds() / 60

        if elapsed_minutes > interval:
            return f"🚨 Due for re-assessment! ({int(elapsed_minutes)}m ago)", "#c0392b"
        else:
            remaining = int(interval - elapsed_minutes)
            return f"✅ Re-assessment due in {remaining}m", "#27ae60"
    except Exception:
        return f"Last reviewed: {last_review}", "gray"


def call_model(patient):
    data = {
        "age": float(patient["age"]),
        "gender": patient.get("gender", "M"),
        "heart_rate_mean": float(patient["heart_rate_mean"]),
        "heart_rate_min": float(patient["heart_rate_min"]),
        "heart_rate_max": float(patient["heart_rate_max"]),
        "systolic_bp_mean": float(patient["systolic_bp_mean"]),
        "systolic_bp_min": float(patient["systolic_bp_min"]),
        "systolic_bp_max": float(patient["systolic_bp_max"]),
        "diastolic_bp_mean": float(patient["diastolic_bp_mean"]),
        "diastolic_bp_min": float(patient["diastolic_bp_min"]),
        "diastolic_bp_max": float(patient["diastolic_bp_max"]),
        "admit_hour": int(patient.get("admit_hour", 12)),
        "admit_dayofweek": int(patient.get("admit_dayofweek", 2)),
        "lactate_max": patient.get("lactate_max"),
        "creatinine_max": patient.get("creatinine_max"),
    }

    feature_dict, computed = compute_features(data)
    X = pd.DataFrame([feature_dict])[FEATURES]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURES)
    risk_score = float(model.predict_proba(X_scaled)[0, 1])
    prediction = risk_score >= THRESHOLD

    if risk_score >= 0.6:
        risk_level, risk_color = "CRITICAL", "#c0392b"
    elif risk_score >= THRESHOLD:
        risk_level, risk_color = "HIGH", "#e67e22"
    elif risk_score >= 0.25:
        risk_level, risk_color = "MODERATE", "#f1c40f"
    else:
        risk_level, risk_color = "LOW", "#27ae60"

    risk_drivers = get_risk_drivers(feature_dict, X_scaled)
    alerts = generate_alerts(data, computed)

    result = {
        "risk_score": round(risk_score, 4),
        "risk_percent": f"{risk_score * 100:.1f}%",
        "risk_level": risk_level,
        "risk_color": risk_color,
        "threshold": THRESHOLD,
        "prediction": "At Risk of Deterioration" if prediction else "Low Risk",
        "is_at_risk": prediction,
        "risk_drivers": risk_drivers,
        "clinical_alerts": alerts,
        "computed_indicators": computed,
    }

    patient["model_result"] = result
    return result


REQUIRED_MODEL_FIELDS = [
    "age", "heart_rate_mean", "heart_rate_min", "heart_rate_max",
    "systolic_bp_mean", "systolic_bp_min", "systolic_bp_max",
    "diastolic_bp_mean", "diastolic_bp_min", "diastolic_bp_max"
]


def get_missing_model_fields(patient):
    missing = []
    for field in REQUIRED_MODEL_FIELDS:
        value = patient.get(field)
        if value is None:
            missing.append(field)
            continue
        try:
            if pd.isna(value):
                missing.append(field)
        except Exception:
            pass
    return missing


def calculate_risk_score(patient):
    if "_risk_score" in patient and "_risk_level" in patient:
        return int(patient["_risk_score"] * 100), patient["_risk_level"]

    missing_fields = get_missing_model_fields(patient)
    if missing_fields:
        patient["_risk_level"] = "INCOMPLETE"
        patient["_risk_score"] = 0
        patient["_missing_model_fields"] = missing_fields
        return 0, "INCOMPLETE"

    try:
        result = call_model(patient)
        score = float(result["risk_score"])
        level = result["risk_level"]
        patient["_risk_score"] = score
        patient["_risk_level"] = level
        return int(score * 100), level
    except Exception as e:
        patient["_risk_level"] = "ERROR"
        patient["_risk_score"] = 0
        return 0, "ERROR"


def batch_predict_patients(patient_list):
    if not patient_list:
        return

    feature_rows = []
    for patient in patient_list:
        data = {
            "age": float(patient["age"]),
            "gender": patient.get("gender", "M"),
            "heart_rate_mean": float(patient["heart_rate_mean"]),
            "heart_rate_min": float(patient["heart_rate_min"]),
            "heart_rate_max": float(patient["heart_rate_max"]),
            "systolic_bp_mean": float(patient["systolic_bp_mean"]),
            "systolic_bp_min": float(patient["systolic_bp_min"]),
            "systolic_bp_max": float(patient["systolic_bp_max"]),
            "diastolic_bp_mean": float(patient["diastolic_bp_mean"]),
            "diastolic_bp_min": float(patient["diastolic_bp_min"]),
            "diastolic_bp_max": float(patient["diastolic_bp_max"]),
            "admit_hour": int(patient.get("admit_hour", 12)),
            "admit_dayofweek": int(patient.get("admit_dayofweek", 2)),
            "lactate_max": patient.get("lactate_max"),
            "creatinine_max": patient.get("creatinine_max")
        }
        feature_dict, _ = compute_features(data)
        feature_rows.append(feature_dict)

    X = pd.DataFrame(feature_rows)[FEATURES]
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    for patient, risk_score in zip(patient_list, probabilities):
        risk_score = float(risk_score)
        if risk_score >= 0.6:
            level = "CRITICAL"
        elif risk_score >= THRESHOLD:
            level = "HIGH"
        elif risk_score >= 0.25:
            level = "MODERATE"
        else:
            level = "LOW"

        patient["_risk_score"] = risk_score
        patient["_risk_level"] = level


def get_risk_color(level):
    colors = {
        "CRITICAL": "#c0392b",
        "HIGH": "#e67e22",
        "MODERATE": "#f1c40f",
        "LOW": "#27ae60",
        "INCOMPLETE": "#7f8c8d",
        "ERROR": "#7f8c8d"
    }
    return colors.get(level, "#7f8c8d")


def get_recommended_actions(level):
    if level == "CRITICAL":
        return [
            "Immediate physician review",
            "Repeat vital signs urgently",
            "Review lactate / creatinine results",
            "Consider escalation to senior clinician"
        ]
    if level == "HIGH":
        return [
            "Close monitoring",
            "Recheck blood pressure and heart rate",
            "Review recent lab trends",
            "Reassess within 30 minutes"
        ]
    if level == "MODERATE":
        return [
            "Continue monitoring",
            "Repeat assessment later",
            "Check for missing lab values"
        ]
    if level == "LOW":
        return [
            "Continue routine monitoring",
            "No urgent action required"
        ]
    return ["Model unavailable - check backend connection"]


class LoginWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ICU Clinical Decision Support System - Login")
        self.geometry("500x520")
        self.resizable(False, False)
        self.logged_user = None

        container = ctk.CTkFrame(self, width=400, height=400)
        container.pack(expand=True, padx=40, pady=40)

        ctk.CTkLabel(
            container,
            text="ICU Clinical Decision\nSupport System",
            font=("Arial", 27, "bold")
        ).pack(pady=(35, 10))

        ctk.CTkLabel(
            container,
            text="Healthcare Professional Login",
            font=("Arial", 17)
        ).pack(pady=(0, 30))

        self.username_entry = ctk.CTkEntry(
            container,
            width=300,
            height=42,
            placeholder_text="Username"
        )
        self.username_entry.pack(pady=10)

        self.password_entry = ctk.CTkEntry(
            container,
            width=300,
            height=42,
            placeholder_text="Password",
            show="*"
        )
        self.password_entry.pack(pady=10)

        ctk.CTkButton(
            container,
            text="Login",
            width=300,
            height=42,
            command=self.login
        ).pack(pady=25)

        self.status_label = ctk.CTkLabel(
            container,
            text="",
            text_color="red"
        )
        self.status_label.pack()

        self.password_entry.bind(
            "<Return>",
            lambda event: self.login()
        )

    def login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get()

        user = USERS.get(username)

        if user is None:
            self.status_label.configure(
                text="Invalid username or password."
            )
            return

        password_hash = hash_password(password)

        if password_hash != user["password"]:
            self.status_label.configure(
                text="Invalid username or password."
            )
            return

        self.logged_user = {
            "username": username,
            "name": user["name"],
            "role": user["role"]
        }

        self.quit()


def save_patients_bulk(patient_list):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    data = []
    for patient in patient_list:
        data.append((
            str(patient.get("subject_id", patient.get("id", ""))),
            str(patient.get("hadm_id", "")),
            str(patient.get("icustay_id", "")),
            patient.get("intime", ""),
            patient.get("gender"),
            patient.get("age"),
            patient.get("heart_rate_mean"),
            patient.get("heart_rate_min"),
            patient.get("heart_rate_max"),
            patient.get("systolic_bp_mean"),
            patient.get("systolic_bp_min"),
            patient.get("systolic_bp_max"),
            patient.get("diastolic_bp_mean"),
            patient.get("diastolic_bp_min"),
            patient.get("diastolic_bp_max"),
            patient.get("creatinine_max"),
            patient.get("lactate_max"),
            patient.get("admit_hour", 12),
            patient.get("admit_dayofweek", 2),
            str(patient.get("icu_hours", "N/A")),
            patient.get("admission_type", ""),
            patient.get("diagnosis", ""),
            patient.get("alert_status", "New"),
            patient.get("clinical_note", ""),
            patient.get("hospital_expire_flag"),
            patient.get("_risk_score"),
            patient.get("_risk_level"),
            patient.get("room_number")
        ))

    cursor.executemany("""
        INSERT OR REPLACE INTO patients (
            subject_id, hadm_id, icustay_id, intime, gender, age,
            heart_rate_mean, heart_rate_min, heart_rate_max,
            systolic_bp_mean, systolic_bp_min, systolic_bp_max,
            diastolic_bp_mean, diastolic_bp_min, diastolic_bp_max,
            creatinine_max, lactate_max, admit_hour, admit_dayofweek,
            icu_hours, admission_type, diagnosis, alert_status,
            clinical_note, hospital_expire_flag, risk_score, risk_level, room_number
        )
        VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?, ?
        )
    """, data)

    connection.commit()
    connection.close()


class ICUApp(ctk.CTk):
    def __init__(self, logged_user):
        super().__init__()

        self.logged_user = logged_user

        self.title("ICU Clinical Decision Support System")
        self.geometry("1350x800")
        self.selected_patient = patients[0] if patients else None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.main = ctk.CTkScrollableFrame(self)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)

        self.create_sidebar()
        self.show_dashboard()

    def create_sidebar(self):
        ctk.CTkLabel(
            self.sidebar,
            text="🏥 ICU Decision\nSupport System",
            font=("Arial", 20, "bold")
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            self.sidebar,
            text=f"{self.logged_user['name']}\n({self.logged_user['role']})",
            font=("Arial", 12),
            text_color="gray"
        ).pack(pady=(0, 10))

        sections = [
            (
                "OVERVIEW & TRIAGE",
                [
                    ("Dashboard", "📊  Dashboard", self.show_dashboard),
                    ("Priority Queue", "🚨  Priority Queue", self.show_priority_queue),
                    ("Alerts Workflow", "🔔  Alerts Workflow", self.show_alerts),
                ]
            ),
            (
                "PATIENT MANAGEMENT",
                [
                    ("Add Patient", "➕  Add Patient", self.show_add_patient),
                    ("Upload Patients CSV", "📁  Upload Patients CSV", self.upload_csv),
                ]
            ),
            (
                "PATIENT CLINICAL VIEW",
                [
                    ("Patient Summary", "📋  Patient Summary", self.show_patient_summary),
                    ("Patient Details", "👤  Patient Details", self.show_patient_details),
                    ("Recommended Actions", "🩺  Recommended Actions", self.show_recommended_actions),
                    ("Risk Explanation", "💡  Risk Explanation", self.show_risk_explanation),
                ]
            ),
            (
                "ANALYTICS & TOOLS",
                [
                    ("What-if Analysis", "🧪  What-if Analysis", self.show_what_if),
                    ("Trend Monitoring", "📈  Trend Monitoring", self.show_trends),
                    ("Missing Data", "⚠️  Missing Data", self.show_missing_data),
                ]
            ),
        ]

        self.sidebar_buttons = {}

        for section_title, buttons in sections:
            ctk.CTkLabel(
                self.sidebar,
                text=section_title,
                font=("Arial", 10, "bold"),
                text_color="#7f8c8d",
                anchor="w"
            ).pack(fill="x", padx=16, pady=(8, 2))

            for key, text, command in buttons:
                btn = ctk.CTkButton(
                    self.sidebar,
                    text=text,
                    anchor="w",
                    height=28,
                    font=("Arial", 12),
                    command=lambda k=key, cmd=command: self.on_button_click(k, cmd)
                )
                btn.pack(pady=2, fill="x", padx=12)
                self.sidebar_buttons[key] = btn

        self.set_active_button("Dashboard")

    def edit_patient_room(self, patient):
        user_role = getattr(self, "logged_user", {}).get("role", "")
        if user_role != "Doctor":
            messagebox.showwarning("Permission Denied",
                                   "Only Doctors are authorized to update patient room assignments.")
            return

        dialog = ctk.CTkInputDialog(
            text=f"Enter new room assignment for Patient {patient.get('id')}:",
            title="Assign / Edit Room"
        )
        new_room = dialog.get_input()

        if new_room and new_room.strip():
            updated_room = new_room.strip()
            patient["room_number"] = updated_room
            self.selected_patient = patient

            for p in patients:
                if str(p.get("subject_id", p.get("id"))) == str(patient.get("subject_id", patient.get("id"))):
                    p["room_number"] = updated_room
                    break

            save_patient_to_database(patient)

            if hasattr(self, "room_label_var") and self.room_label_var:
                self.room_label_var.configure(text=f"Room Assignment: {updated_room}")
            else:
                self.show_patient_summary()

            messagebox.showinfo("Room Updated",
                                f"Patient {patient.get('id')} room successfully set to {updated_room}.")

    def edit_clinical_note(self, patient):
        user_role = getattr(self, "logged_user", {}).get("role", "")
        if user_role != "Doctor":
            messagebox.showwarning("Permission Denied", "Only Doctors are authorized to enter clinical justification notes.")
            return

        dialog = ctk.CTkInputDialog(
            text=f"Enter Physician Note / Override Justification for Patient {patient.get('id')}:",
            title="Physician Clinical Note"
        )
        new_note = dialog.get_input()

        if new_note is not None:
            updated_note = new_note.strip()
            patient["clinical_note"] = updated_note
            self.selected_patient = patient

            for p in patients:
                if str(p.get("subject_id", p.get("id"))) == str(patient.get("subject_id", patient.get("id"))):
                    p["clinical_note"] = updated_note
                    break

            save_patient_to_database(patient)

            if hasattr(self, "note_label_var") and self.note_label_var:
                self.note_label_var.configure(
                    text=f"Physician Note: {updated_note or 'None'}",
                    text_color="#2c3e50" if updated_note else "gray"
                )
            else:
                self.show_patient_summary()

            messagebox.showinfo("Note Saved", "Physician justification note successfully attached to patient file.")

    def on_button_click(self, button_key, command):
        self.set_active_button(button_key)
        command()

    def set_active_button(self, active_key):
        ACTIVE_COLOR = "#1f538d"
        DEFAULT_COLOR = "#3b8ed0"

        for key, button in self.sidebar_buttons.items():
            if key == active_key:
                button.configure(fg_color=ACTIVE_COLOR)
            else:
                button.configure(fg_color=DEFAULT_COLOR)

    def require_selected_patient(self):
        if self.selected_patient is None:
            messagebox.showwarning(
                "No Patient Selected",
                "There are currently no patients in the system.\nPlease add a patient or upload a CSV file first."
            )
            return False
        return True

    def clear_main(self):
        for widget in self.main.winfo_children():
            widget.destroy()

    def show_title(self, text):
        ctk.CTkLabel(
            self.main,
            text=text,
            font=("Arial", 28, "bold")
        ).pack(pady=20)

    def select_patient(self, patient):
        self.selected_patient = patient
        self.set_active_button("Patient Summary")
        self.show_patient_summary()

    def show_dashboard(self):
        self.clear_main()
        self.show_title("ICU Dashboard")

        if not patients:
            ctk.CTkLabel(self.main, text="No patients are currently stored in the system.", font=("Arial", 18, "bold")).pack(pady=(40, 10))
            ctk.CTkLabel(self.main, text="Add a patient manually or upload a CSV file to get started.", font=("Arial", 15), text_color="gray").pack(pady=5)
            ctk.CTkButton(self.main, text="Add Patient", width=180, command=self.show_add_patient).pack(pady=15)
            ctk.CTkButton(self.main, text="Upload Patients CSV", width=180, command=self.upload_csv).pack(pady=5)
            return

        if not hasattr(self, "dashboard_page"):
            self.dashboard_page = 0
        if not hasattr(self, "dashboard_search"):
            self.dashboard_search = ""
        if not hasattr(self, "dashboard_search_field"):
            self.dashboard_search_field = "Patient ID"

        PAGE_SIZE = 50

        search_frame = ctk.CTkFrame(self.main)
        search_frame.pack(fill="x", padx=20, pady=(5, 10))

        ctk.CTkLabel(search_frame, text="Search by:", font=("Arial", 15, "bold")).pack(side="left", padx=(15, 8), pady=12)

        search_field_var = ctk.StringVar(value=self.dashboard_search_field)
        search_field_menu = ctk.CTkOptionMenu(
            search_frame,
            values=["Patient ID", "Hospital Admission ID", "ICU Stay ID"],
            variable=search_field_var,
            width=190
        )
        search_field_menu.pack(side="left", padx=5)

        search_var = ctk.StringVar(value=self.dashboard_search)
        search_entry = ctk.CTkEntry(search_frame, width=260, textvariable=search_var, placeholder_text="Enter ID...")
        search_entry.pack(side="left", padx=5)

        def apply_search():
            self.dashboard_search = search_var.get().strip()
            self.dashboard_search_field = search_field_var.get()
            self.dashboard_page = 0
            self.show_dashboard()

        def clear_search():
            self.dashboard_search = ""
            self.dashboard_page = 0
            self.show_dashboard()

        ctk.CTkButton(search_frame, text="Search", width=90, command=apply_search).pack(side="left", padx=5)
        ctk.CTkButton(search_frame, text="Clear", width=80, command=clear_search).pack(side="left", padx=5)
        search_entry.bind("<Return>", lambda event: apply_search())

        query = self.dashboard_search.strip()
        if query:
            field_mapping = {
                "Patient ID": "subject_id",
                "Hospital Admission ID": "hadm_id",
                "ICU Stay ID": "icustay_id"
            }
            selected_field = field_mapping[self.dashboard_search_field]
            filtered_patients = []
            for patient in patients:
                value = patient.get("subject_id", patient.get("id", "")) if selected_field == "subject_id" else patient.get(selected_field, "")
                if str(value).strip() == query:
                    filtered_patients.append(patient)
        else:
            filtered_patients = patients

        total_patients = len(filtered_patients)
        total_pages = max(1, (total_patients + PAGE_SIZE - 1) // PAGE_SIZE)
        if self.dashboard_page >= total_pages:
            self.dashboard_page = total_pages - 1

        start_index = self.dashboard_page * PAGE_SIZE
        end_index = start_index + PAGE_SIZE
        page_patients = filtered_patients[start_index:end_index]

        info_text = f"Showing {start_index + 1:,}–{min(end_index, total_patients):,} of {total_patients:,} patients" if total_patients else "No patients found."
        ctk.CTkLabel(self.main, text=info_text, font=("Arial", 13), text_color="gray").pack(anchor="w", padx=25, pady=(0, 5))

        frame = ctk.CTkFrame(self.main)
        frame.pack(fill="x", padx=20, pady=10)

        headers = ["Patient ID", "Room", "Age", "Gender", "ICU Hours", "Risk Score", "Risk Level", "Action"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(frame, text=header, font=("Arial", 14, "bold")).grid(row=0, column=col, padx=12, pady=10)

        if not page_patients:
            ctk.CTkLabel(frame, text="No patients match your search.", font=("Arial", 16)).grid(row=1, column=0, columnspan=len(headers), pady=30)

        for row, patient in enumerate(page_patients, start=1):
            score, level = calculate_risk_score(patient)
            patient_id = patient.get("subject_id", patient.get("id", "N/A"))
            room_val = patient.get("room_number", f"ICU-{(row % 15) + 101}")

            values = [patient_id, room_val, patient.get("age", "N/A"), patient.get("gender", "N/A"), patient.get("icu_hours", "N/A"), f"{score}/100", level]

            for col, value in enumerate(values):
                if col == 6:
                    ctk.CTkLabel(frame, text=value, fg_color=get_risk_color(level), text_color="white", corner_radius=8, width=100).grid(row=row, column=col, padx=12, pady=8)
                else:
                    ctk.CTkLabel(frame, text=str(value), font=("Arial", 13)).grid(row=row, column=col, padx=12, pady=8)

            ctk.CTkButton(frame, text="Open", width=80, command=lambda p=patient: self.select_patient(p)).grid(row=row, column=7, padx=12, pady=8)

        pagination_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        pagination_frame.pack(fill="x", padx=20, pady=15)

        def previous_page():
            if self.dashboard_page > 0:
                self.dashboard_page -= 1
                self.show_dashboard()

        def next_page():
            if self.dashboard_page < total_pages - 1:
                self.dashboard_page += 1
                self.show_dashboard()

        previous_button = ctk.CTkButton(pagination_frame, text="← Previous", width=120, command=previous_page)
        previous_button.pack(side="left", padx=10)
        if self.dashboard_page == 0:
            previous_button.configure(state="disabled")

        ctk.CTkLabel(pagination_frame, text=f"Page {self.dashboard_page + 1} of {total_pages}", font=("Arial", 14, "bold")).pack(side="left", expand=True)

        next_button = ctk.CTkButton(pagination_frame, text="Next →", width=120, command=next_page)
        next_button.pack(side="right", padx=10)
        if self.dashboard_page >= total_pages - 1:
            next_button.configure(state="disabled")

    def show_priority_queue(self):
        self.clear_main()
        self.show_title("Clinical Priority Queue")

        if not hasattr(self, "priority_page"):
            self.priority_page = 0
        if not hasattr(self, "priority_filter"):
            self.priority_filter = "All"

        PAGE_SIZE = 25

        filter_frame = ctk.CTkFrame(self.main)
        filter_frame.pack(fill="x", padx=25, pady=(5, 15))

        ctk.CTkLabel(filter_frame, text="Show:", font=("Arial", 14, "bold")).pack(side="left", padx=(15, 8), pady=12)

        filter_var = ctk.StringVar(value=self.priority_filter)
        filter_menu = ctk.CTkOptionMenu(filter_frame, values=["All", "CRITICAL", "HIGH", "MODERATE", "LOW"], variable=filter_var, width=150)
        filter_menu.pack(side="left", padx=5)

        def apply_filter():
            self.priority_filter = filter_var.get()
            self.priority_page = 0
            self.show_priority_queue()

        ctk.CTkButton(filter_frame, text="Apply", width=90, command=apply_filter).pack(side="left", padx=10)

        ranked = []
        for patient in patients:
            if get_missing_model_fields(patient):
                continue
            if "_risk_score" in patient and "_risk_level" in patient:
                score = int(float(patient["_risk_score"]) * 100)
                level = patient["_risk_level"]
            else:
                score, level = calculate_risk_score(patient)

            if level in ["ERROR", "INCOMPLETE"]:
                continue
            if self.priority_filter != "All" and level != self.priority_filter:
                continue

            ranked.append((score, level, patient))

        ranked.sort(key=lambda item: item[0], reverse=True)

        total_patients = len(ranked)
        total_pages = max(1, (total_patients + PAGE_SIZE - 1) // PAGE_SIZE)
        if self.priority_page >= total_pages:
            self.priority_page = total_pages - 1

        start = self.priority_page * PAGE_SIZE
        end = start + PAGE_SIZE
        page_patients = ranked[start:end]

        info_text = f"Showing {start + 1:,}–{min(end, total_patients):,} of {total_patients:,} patients" if total_patients else "No patients match the selected filter."
        ctk.CTkLabel(self.main, text=info_text, font=("Arial", 13), text_color="gray").pack(anchor="w", padx=30, pady=(0, 10))

        for index, (score, level, patient) in enumerate(page_patients, start=start + 1):
            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=8)

            patient_id = patient.get("subject_id", patient.get("id", "N/A"))
            room_num = patient.get("room_number", "Unassigned")

            ctk.CTkLabel(card, text=f"{index}. Patient {patient_id} ({room_num}) — {level} risk — {score}/100", font=("Arial", 19, "bold"), text_color=get_risk_color(level)).pack(anchor="w", padx=20, pady=(10, 5))

            actions = get_recommended_actions(level)
            if actions:
                ctk.CTkLabel(card, text=f"Suggested priority: {actions[0]}", font=("Arial", 14)).pack(anchor="w", padx=20, pady=4)

            ctk.CTkButton(card, text="Open Patient", width=120, command=lambda p=patient: self.select_patient(p)).pack(anchor="e", padx=20, pady=10)

        pagination = ctk.CTkFrame(self.main, fg_color="transparent")
        pagination.pack(fill="x", padx=25, pady=20)

        def previous_page():
            if self.priority_page > 0:
                self.priority_page -= 1
                self.show_priority_queue()

        def next_page():
            if self.priority_page < total_pages - 1:
                self.priority_page += 1
                self.show_priority_queue()

        previous_button = ctk.CTkButton(pagination, text="← Previous", width=120, command=previous_page)
        previous_button.pack(side="left")
        if self.priority_page == 0:
            previous_button.configure(state="disabled")

        ctk.CTkLabel(pagination, text=f"Page {self.priority_page + 1} of {total_pages}", font=("Arial", 14, "bold")).pack(side="left", expand=True)

        next_button = ctk.CTkButton(pagination, text="Next →", width=120, command=next_page)
        next_button.pack(side="right")
        if self.priority_page >= total_pages - 1:
            next_button.configure(state="disabled")

    def show_patient_summary(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient
        score, level = calculate_risk_score(patient)
        result = patient.get("model_result", {})

        self.show_title(f"Patient Summary Card - {patient['id']}")

        card = ctk.CTkFrame(self.main)
        card.pack(fill="x", padx=25, pady=15)

        # 1. חדר
        room_num = patient.get("room_number", "Unassigned")
        room_frame = ctk.CTkFrame(card, fg_color="transparent")
        room_frame.pack(anchor="w", padx=25, pady=(12, 4))

        self.room_label_var = ctk.CTkLabel(room_frame, text=f"Room Assignment: {room_num}", font=("Arial", 17, "bold"))
        self.room_label_var.pack(side="left")

        ctk.CTkButton(
            room_frame,
            text="✏️ Edit Room",
            width=100,
            height=26,
            font=("Arial", 12),
            command=lambda: self.edit_patient_room(patient)
        ).pack(side="left", padx=15)

        # 2. סיכום מדדים
        summary = [
            f"Risk level: {level}",
            f"Risk score: {score}/100",
            f"Prediction: {result.get('prediction', 'N/A')}",
            f"Admission type: {patient.get('admission_type', 'N/A')}",
            f"Diagnosis: {patient.get('diagnosis', 'N/A')}",
            f"Alert status: {patient.get('alert_status', 'New')}",
        ]

        for line in summary:
            ctk.CTkLabel(card, text=line, font=("Arial", 17)).pack(anchor="w", padx=25, pady=5)

        # 3. הערת רופא / נימוק קליני (Physician Override Note)
        note_text = patient.get("clinical_note", "") or "None"
        note_frame = ctk.CTkFrame(card, fg_color="transparent")
        note_frame.pack(anchor="w", padx=25, pady=(8, 8))

        self.note_label_var = ctk.CTkLabel(
            note_frame,
            text=f"Physician Note: {note_text}",
            font=("Arial", 16, "italic"),
            text_color="#2c3e50" if note_text != "None" else "gray"
        )
        self.note_label_var.pack(side="left")

        ctk.CTkButton(
            note_frame,
            text="📝 Edit Physician Note",
            width=160,
            height=26,
            font=("Arial", 12),
            command=lambda: self.edit_clinical_note(patient)
        ).pack(side="left", padx=15)

        # 4. המלצות
        ctk.CTkLabel(card, text="Suggested next steps:", font=("Arial", 18, "bold")).pack(anchor="w", padx=25, pady=(15, 8))

        for action in get_recommended_actions(level):
            ctk.CTkLabel(card, text=f"• {action}", font=("Arial", 15)).pack(anchor="w", padx=45, pady=3)

    # Patient Details עם תמיכה מלאה בעריכת מדדים קליניים
    def show_patient_details(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient
        self.show_title(f"Patient Details - {patient['id']}")

        # סרגל עליון עם כפתור עריכה
        top_bar = ctk.CTkFrame(self.main, fg_color="transparent")
        top_bar.pack(fill="x", padx=25, pady=(0, 10))

        ctk.CTkButton(
            top_bar,
            text="✏️ Edit Patient Vitals & Labs",
            width=220,
            height=32,
            font=("Arial", 13, "bold"),
            command=lambda: self.edit_patient_vitals(patient)
        ).pack(side="right")

        details = [
            ("Room Assignment", patient.get("room_number", "Unassigned")),
            ("Age", patient["age"]),
            ("Gender", patient["gender"]),
            ("ICU Hours", patient.get("icu_hours", "N/A")),
            ("Heart Rate Mean", patient["heart_rate_mean"]),
            ("Heart Rate Min", patient["heart_rate_min"]),
            ("Heart Rate Max", patient["heart_rate_max"]),
            ("Systolic BP Mean", patient["systolic_bp_mean"]),
            ("Systolic BP Min", patient["systolic_bp_min"]),
            ("Systolic BP Max", patient["systolic_bp_max"]),
            ("Diastolic BP Mean", patient["diastolic_bp_mean"]),
            ("Diastolic BP Min", patient["diastolic_bp_min"]),
            ("Diastolic BP Max", patient["diastolic_bp_max"]),
            ("Lactate Max", patient.get("lactate_max", "Missing")),
            ("Creatinine Max", patient.get("creatinine_max", "Missing")),
        ]

        frame = ctk.CTkFrame(self.main)
        frame.pack(fill="x", padx=25, pady=15)

        for key, value in details:
            ctk.CTkLabel(frame, text=f"{key}: {value}", font=("Arial", 16)).pack(anchor="w", padx=25, pady=5)

    def edit_patient_vitals(self, patient):
        """חלון עריכת מדדים פיזיולוגיים ומעבדה + חישוב סיכון מחדש"""
        user_role = getattr(self, "logged_user", {}).get("role", "")
        if user_role != "Doctor":
            messagebox.showwarning("Permission Denied", "Only Doctors are authorized to edit patient vital measurements.")
            return

        edit_win = ctk.CTkToplevel(self)
        edit_win.title(f"Edit Vitals - Patient {patient.get('id')}")
        edit_win.geometry("520x650")
        edit_win.resizable(False, False)
        edit_win.transient(self)

        ctk.CTkLabel(edit_win, text=f"Edit Vitals & Labs (Patient {patient.get('id')})", font=("Arial", 20, "bold")).pack(pady=15)

        scroll_form = ctk.CTkScrollableFrame(edit_win, width=460, height=480)
        scroll_form.pack(padx=20, pady=10)

        fields_to_edit = [
            ("age", "Age", patient.get("age")),
            ("heart_rate_mean", "Heart Rate Mean", patient.get("heart_rate_mean")),
            ("heart_rate_min", "Heart Rate Min", patient.get("heart_rate_min")),
            ("heart_rate_max", "Heart Rate Max", patient.get("heart_rate_max")),
            ("systolic_bp_mean", "Systolic BP Mean", patient.get("systolic_bp_mean")),
            ("systolic_bp_min", "Systolic BP Min", patient.get("systolic_bp_min")),
            ("systolic_bp_max", "Systolic BP Max", patient.get("systolic_bp_max")),
            ("diastolic_bp_mean", "Diastolic BP Mean", patient.get("diastolic_bp_mean")),
            ("diastolic_bp_min", "Diastolic BP Min", patient.get("diastolic_bp_min")),
            ("diastolic_bp_max", "Diastolic BP Max", patient.get("diastolic_bp_max")),
            ("lactate_max", "Lactate Max", "" if patient.get("lactate_max") is None else patient.get("lactate_max")),
            ("creatinine_max", "Creatinine Max", "" if patient.get("creatinine_max") is None else patient.get("creatinine_max")),
        ]

        entries = {}
        for key, label_text, val in fields_to_edit:
            row_frame = ctk.CTkFrame(scroll_form, fg_color="transparent")
            row_frame.pack(fill="x", pady=5, padx=10)

            ctk.CTkLabel(row_frame, text=label_text, width=180, anchor="w", font=("Arial", 14)).pack(side="left")
            entry = ctk.CTkEntry(row_frame, width=180)
            entry.pack(side="right")
            if val is not None and str(val) != "":
                entry.insert(0, str(val))
            entries[key] = entry

        def save_vitals():
            try:
                for key, _, _ in fields_to_edit:
                    raw_val = entries[key].get().strip()
                    if raw_val == "" or raw_val.lower() == "none" or raw_val.lower() == "nan":
                        patient[key] = None
                    else:
                        patient[key] = float(raw_val)

                # איפוס מטמון לחישוב מחדש של המודל
                patient.pop("_risk_score", None)
                patient.pop("_risk_level", None)
                patient.pop("model_result", None)

                call_model(patient)
                calculate_risk_score(patient)

                self.selected_patient = patient
                for p in patients:
                    if str(p.get("subject_id", p.get("id"))) == str(patient.get("subject_id", patient.get("id"))):
                        p.update(patient)
                        break

                save_patient_to_database(patient)

                edit_win.destroy()
                self.show_patient_details()

                messagebox.showinfo(
                    "Vitals Updated",
                    f"Patient {patient.get('id')} vitals updated successfully.\n\n"
                    f"New Risk Level: {patient.get('_risk_level')}\n"
                    f"New Risk Score: {int(patient.get('_risk_score', 0) * 100)}/100"
                )

            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values for vitals.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not update vitals:\n{e}")

        ctk.CTkButton(
            edit_win,
            text="Save Changes & Recalculate Risk",
            width=260,
            height=38,
            font=("Arial", 14, "bold"),
            command=save_vitals
        ).pack(pady=15)

    def show_risk_explanation(self):
        if not self.require_selected_patient():
            return

        self.clear_main()
        patient = self.selected_patient
        self.show_title(f"Why This Risk Score? - {patient['id']}")

        result = patient.get("model_result")
        if not result or not result.get("risk_drivers"):
            missing_fields = get_missing_model_fields(patient)
            if missing_fields:
                ctk.CTkLabel(self.main, text="Risk explanation is unavailable because required clinical data is missing.", font=("Arial", 17, "bold")).pack(pady=(25, 10))
                ctk.CTkLabel(self.main, text="Missing fields:", font=("Arial", 15)).pack(pady=5)
                for field in missing_fields:
                    ctk.CTkLabel(self.main, text=f"• {field.replace('_', ' ').title()}", font=("Arial", 14)).pack(pady=2)
                return

            try:
                result = call_model(patient)
            except Exception as e:
                ctk.CTkLabel(self.main, text=f"Could not generate risk explanation:\n{e}", font=("Arial", 16), text_color="#c0392b").pack(pady=25)
                return

        drivers = result.get("risk_drivers", [])
        if not drivers:
            ctk.CTkLabel(self.main, text="No significant risk drivers were identified for this patient.", font=("Arial", 17)).pack(pady=25)
            return

        ctk.CTkLabel(self.main, text="Main factors influencing this patient's risk:", font=("Arial", 18, "bold")).pack(pady=(10, 15))

        for driver in drivers:
            direction = driver.get("direction")
            arrow = "▲ Increases risk" if direction == "increase" else "▼ Decreases risk"
            color = "#c0392b" if direction == "increase" else "#27ae60"

            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=40, pady=8)

            ctk.CTkLabel(card, text=f"{driver.get('name', 'Unknown factor')} — {driver.get('value', 'N/A')}", font=("Arial", 18, "bold")).pack(anchor="w", padx=20, pady=(10, 4))
            ctk.CTkLabel(card, text=arrow, text_color=color, font=("Arial", 15, "bold")).pack(anchor="w", padx=20, pady=3)
            ctk.CTkLabel(card, text=f"Model impact: {driver.get('impact', 'N/A')}", font=("Arial", 13), text_color="gray").pack(anchor="w", padx=20, pady=(3, 10))

    def show_recommended_actions(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient
        score, level = calculate_risk_score(patient)

        self.show_title(f"Recommended Actions - {patient['id']}")
        ctk.CTkLabel(self.main, text=f"Current risk level: {level} ({score}/100)", font=("Arial", 22, "bold"), text_color=get_risk_color(level)).pack(pady=10)

        for action in get_recommended_actions(level):
            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=8)
            ctk.CTkLabel(card, text=f"• {action}", font=("Arial", 17)).pack(anchor="w", padx=25, pady=12)

        ctk.CTkLabel(self.main, text="Note: recommendations are for decision support only and do not replace clinical judgment.", text_color="gray", font=("Arial", 14)).pack(pady=20)

    def show_alerts(self):
        self.clear_main()
        self.show_title("Alert Workflow")

        if not hasattr(self, "alerts_page"):
            self.alerts_page = 0
        if not hasattr(self, "alerts_risk_filter"):
            self.alerts_risk_filter = "All"
        if not hasattr(self, "alerts_status_filter"):
            self.alerts_status_filter = "Active"

        PAGE_SIZE = 25

        filter_frame = ctk.CTkFrame(self.main)
        filter_frame.pack(fill="x", padx=25, pady=(5, 15))

        ctk.CTkLabel(filter_frame, text="Risk Level:", font=("Arial", 14, "bold")).pack(side="left", padx=(15, 5), pady=12)
        risk_var = ctk.StringVar(value=self.alerts_risk_filter)
        risk_menu = ctk.CTkOptionMenu(filter_frame, values=["All", "CRITICAL", "HIGH", "MODERATE"], variable=risk_var, width=140)
        risk_menu.pack(side="left", padx=5)

        ctk.CTkLabel(filter_frame, text="Status:", font=("Arial", 14, "bold")).pack(side="left", padx=(20, 5))
        status_var = ctk.StringVar(value=self.alerts_status_filter)
        status_menu = ctk.CTkOptionMenu(filter_frame, values=["Active", "All", "New", "Reviewed", "In Progress", "Resolved"], variable=status_var, width=150)
        status_menu.pack(side="left", padx=5)

        def apply_filters():
            self.alerts_risk_filter = risk_var.get()
            self.alerts_status_filter = status_var.get()
            self.alerts_page = 0
            self.show_alerts()

        ctk.CTkButton(filter_frame, text="Apply", width=90, command=apply_filters).pack(side="left", padx=15)

        alert_patients = []
        for patient in patients:
            if get_missing_model_fields(patient):
                continue

            if "_risk_score" in patient and "_risk_level" in patient:
                score = int(patient["_risk_score"] * 100)
                level = patient["_risk_level"]
            else:
                score, level = calculate_risk_score(patient)

            if level not in ["CRITICAL", "HIGH", "MODERATE"]:
                continue

            status = patient.get("alert_status", "New")
            if self.alerts_risk_filter != "All" and level != self.alerts_risk_filter:
                continue

            if self.alerts_status_filter == "Active":
                if status == "Resolved":
                    continue
            elif self.alerts_status_filter != "All" and status != self.alerts_status_filter:
                continue

            alert_patients.append((score, level, patient))

        total_alerts = len(alert_patients)
        total_pages = max(1, (total_alerts + PAGE_SIZE - 1) // PAGE_SIZE)
        if self.alerts_page >= total_pages:
            self.alerts_page = total_pages - 1

        start = self.alerts_page * PAGE_SIZE
        end = start + PAGE_SIZE
        current_page = alert_patients[start:end]

        info = f"Showing {start + 1:,}–{min(end, total_alerts):,} of {total_alerts:,} alerts" if total_alerts else "No alerts match the selected filters."
        ctk.CTkLabel(self.main, text=info, font=("Arial", 13), text_color="gray").pack(anchor="w", padx=30, pady=(0, 10))

        for score, level, patient in current_page:
            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=8)

            patient_id = patient.get("subject_id", patient.get("id", "N/A"))
            room_num = patient.get("room_number", "Unassigned")
            status = patient.get("alert_status", "New")

            ctk.CTkLabel(
                card,
                text=f"Patient {patient_id} ({room_num}) | {level} | Risk {score}/100 | Status: {status}",
                font=("Arial", 18, "bold"),
                text_color=get_risk_color(level)
            ).pack(anchor="w", padx=20, pady=(10, 5))

            result = patient.get("model_result", {})
            alerts = result.get("clinical_alerts", [])

            if alerts:
                for alert in alerts[:3]:
                    ctk.CTkLabel(card, text=f"{alert.get('icon', '')} {alert.get('text', '')}", font=("Arial", 13)).pack(anchor="w", padx=25, pady=2)
            else:
                ctk.CTkLabel(card, text="Open the patient for detailed clinical alert analysis.", font=("Arial", 13), text_color="gray").pack(anchor="w", padx=25, pady=4)

            last_review = patient.get("last_reviewed_at", "Never")
            reviewer = patient.get("reviewed_by", "N/A")
            reassess_text, reassess_color = get_reassessment_info(patient)

            review_info_frame = ctk.CTkFrame(card, fg_color="transparent")
            review_info_frame.pack(fill="x", padx=20, pady=(2, 5))

            ctk.CTkLabel(
                review_info_frame,
                text=f"⏱️ Last Review: {last_review} (by {reviewer})",
                font=("Arial", 13),
                text_color="gray"
            ).pack(side="left", padx=5)

            ctk.CTkLabel(
                review_info_frame,
                text=reassess_text,
                font=("Arial", 13, "bold"),
                text_color=reassess_color
            ).pack(side="right", padx=5)

            button_frame = ctk.CTkFrame(card, fg_color="transparent")
            button_frame.pack(anchor="e", padx=15, pady=10)

            ctk.CTkButton(button_frame, text="Open", width=80, command=lambda p=patient: self.select_patient(p)).pack(side="left", padx=4)
            ctk.CTkButton(button_frame, text="Reviewed", width=95, command=lambda p=patient: self.update_alert_status(p, "Reviewed")).pack(side="left", padx=4)
            ctk.CTkButton(button_frame, text="In Progress", width=105, command=lambda p=patient: self.update_alert_status(p, "In Progress")).pack(side="left", padx=4)
            ctk.CTkButton(button_frame, text="Resolved", width=90, command=lambda p=patient: self.update_alert_status(p, "Resolved")).pack(side="left", padx=4)

        pagination = ctk.CTkFrame(self.main, fg_color="transparent")
        pagination.pack(fill="x", padx=25, pady=20)

        def previous_page():
            if self.alerts_page > 0:
                self.alerts_page -= 1
                self.show_alerts()

        def next_page():
            if self.alerts_page < total_pages - 1:
                self.alerts_page += 1
                self.show_alerts()

        previous_button = ctk.CTkButton(pagination, text="← Previous", width=120, command=previous_page)
        previous_button.pack(side="left")
        if self.alerts_page == 0:
            previous_button.configure(state="disabled")

        ctk.CTkLabel(pagination, text=f"Page {self.alerts_page + 1} of {total_pages}", font=("Arial", 14, "bold")).pack(side="left", expand=True)

        next_button = ctk.CTkButton(pagination, text="Next →", width=120, command=next_page)
        next_button.pack(side="right")
        if self.alerts_page >= total_pages - 1:
            next_button.configure(state="disabled")

    def update_alert_status(self, patient, status):
        patient["alert_status"] = status
        patient["last_reviewed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        patient["reviewed_by"] = getattr(self, "logged_user", {}).get("name", "Staff") if hasattr(self, "logged_user") else "Staff"

        save_patient_to_database(patient)

        messagebox.showinfo(
            "Alert Updated",
            f"Patient {patient.get('id', 'N/A')} alert marked as {status}."
        )

        self.show_alerts()

    def show_what_if(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient
        original_score, original_level = calculate_risk_score(patient)

        self.show_title(f"What-if Analysis - {patient['id']}")
        ctk.CTkLabel(self.main, text=f"Original risk: {original_level} ({original_score}/100)", font=("Arial", 20, "bold")).pack(pady=10)

        form = ctk.CTkFrame(self.main)
        form.pack(padx=25, pady=15)

        fields = {
            "heart_rate_mean": ctk.StringVar(value=str(patient["heart_rate_mean"])),
            "systolic_bp_mean": ctk.StringVar(value=str(patient["systolic_bp_mean"])),
            "diastolic_bp_mean": ctk.StringVar(value=str(patient["diastolic_bp_mean"])),
            "lactate_max": ctk.StringVar(value="" if patient.get("lactate_max") is None else str(patient.get("lactate_max"))),
            "creatinine_max": ctk.StringVar(value="" if patient.get("creatinine_max") is None else str(patient.get("creatinine_max"))),
        }

        for i, (field, var) in enumerate(fields.items()):
            ctk.CTkLabel(form, text=field, font=("Arial", 15)).grid(row=i, column=0, padx=15, pady=8)
            ctk.CTkEntry(form, textvariable=var, width=180).grid(row=i, column=1, padx=15, pady=8)

        result_label = ctk.CTkLabel(self.main, text="", font=("Arial", 20, "bold"))
        result_label.pack(pady=20)

        def run_what_if():
            try:
                simulated = copy.deepcopy(patient)
                simulated["heart_rate_mean"] = float(fields["heart_rate_mean"].get())
                simulated["systolic_bp_mean"] = float(fields["systolic_bp_mean"].get())
                simulated["diastolic_bp_mean"] = float(fields["diastolic_bp_mean"].get())

                simulated["lactate_max"] = None if fields["lactate_max"].get() == "" else float(fields["lactate_max"].get())
                simulated["creatinine_max"] = None if fields["creatinine_max"].get() == "" else float(fields["creatinine_max"].get())

                new_score, new_level = calculate_risk_score(simulated)
                result_label.configure(text=f"New risk: {new_level} ({new_score}/100)")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ctk.CTkButton(self.main, text="Run What-if Analysis", command=run_what_if).pack(pady=10)

    def show_trends(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient

        self.show_title(f"Trend Monitoring - {patient['id']}")

        trends = [
            ("Heart Rate", patient["heart_rate_min"], patient["heart_rate_mean"], patient["heart_rate_max"]),
            ("Systolic BP", patient["systolic_bp_min"], patient["systolic_bp_mean"], patient["systolic_bp_max"]),
            ("Diastolic BP", patient["diastolic_bp_min"], patient["diastolic_bp_mean"], patient["diastolic_bp_max"]),
        ]

        for name, min_v, mean_v, max_v in trends:
            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=10)

            trend_text = f"{name}: min={min_v}, mean={mean_v}, max={max_v}"
            if max_v - min_v > 40:
                interpretation = "High variability — may indicate instability"
                color = "#e67e22"
            else:
                interpretation = "No major variability detected"
                color = "#27ae60"

            ctk.CTkLabel(card, text=trend_text, font=("Arial", 18, "bold")).pack(anchor="w", padx=20, pady=8)
            ctk.CTkLabel(card, text=interpretation, text_color=color, font=("Arial", 15)).pack(anchor="w", padx=20, pady=5)

    def show_add_patient(self):
        self.clear_main()
        self.show_title("Add New Patient")

        form = ctk.CTkFrame(self.main)
        form.pack(fill="x", padx=30, pady=15)

        fields_config = [
            ("subject_id", "Subject ID", ""),
            ("hadm_id", "Hospital Admission ID", ""),
            ("icustay_id", "ICU Stay ID", ""),
            ("age", "Age", ""),
            ("gender", "Gender (M/F)", "M"),
            ("heart_rate_mean", "Heart Rate Mean", ""),
            ("heart_rate_min", "Heart Rate Min", ""),
            ("heart_rate_max", "Heart Rate Max", ""),
            ("systolic_bp_mean", "Systolic BP Mean", ""),
            ("systolic_bp_min", "Systolic BP Min", ""),
            ("systolic_bp_max", "Systolic BP Max", ""),
            ("diastolic_bp_mean", "Diastolic BP Mean", ""),
            ("diastolic_bp_min", "Diastolic BP Min", ""),
            ("diastolic_bp_max", "Diastolic BP Max", ""),
            ("creatinine_max", "Creatinine Max (optional)", ""),
            ("lactate_max", "Lactate Max (optional)", ""),
            ("admit_hour", "Admission Hour (0-23)", "12"),
            ("admit_dayofweek", "Admission Day (0=Mon ... 6=Sun)", "2"),
        ]

        entries = {}
        for i, (key, label_text, default) in enumerate(fields_config):
            row = i // 2
            side = i % 2
            block = ctk.CTkFrame(form, fg_color="transparent")
            block.grid(row=row, column=side, padx=25, pady=7, sticky="ew")

            ctk.CTkLabel(block, text=label_text, width=210, anchor="w").pack(side="left", padx=5)
            entry = ctk.CTkEntry(block, width=180)
            entry.pack(side="left", padx=5)

            if default:
                entry.insert(0, default)
            entries[key] = entry

        def save_patient():
            try:
                required_numeric = [
                    "age", "heart_rate_mean", "heart_rate_min", "heart_rate_max",
                    "systolic_bp_mean", "systolic_bp_min", "systolic_bp_max",
                    "diastolic_bp_mean", "diastolic_bp_min", "diastolic_bp_max"
                ]

                for field in required_numeric:
                    if not entries[field].get().strip():
                        raise ValueError(f"{field.replace('_', ' ').title()} is required.")

                gender = entries["gender"].get().strip().upper()
                if gender not in ["M", "F"]:
                    raise ValueError("Gender must be M or F.")

                patient = {
                    "id": entries["subject_id"].get().strip() or f"P{len(patients) + 1001}",
                    "subject_id": entries["subject_id"].get().strip(),
                    "hadm_id": entries["hadm_id"].get().strip(),
                    "icustay_id": entries["icustay_id"].get().strip(),
                    "age": float(entries["age"].get()),
                    "gender": gender,
                    "heart_rate_mean": float(entries["heart_rate_mean"].get()),
                    "heart_rate_min": float(entries["heart_rate_min"].get()),
                    "heart_rate_max": float(entries["heart_rate_max"].get()),
                    "systolic_bp_mean": float(entries["systolic_bp_mean"].get()),
                    "systolic_bp_min": float(entries["systolic_bp_min"].get()),
                    "systolic_bp_max": float(entries["systolic_bp_max"].get()),
                    "diastolic_bp_mean": float(entries["diastolic_bp_mean"].get()),
                    "diastolic_bp_min": float(entries["diastolic_bp_min"].get()),
                    "diastolic_bp_max": float(entries["diastolic_bp_max"].get()),
                    "creatinine_max": self.optional_float(entries["creatinine_max"].get()),
                    "lactate_max": self.optional_float(entries["lactate_max"].get()),
                    "admit_hour": int(entries["admit_hour"].get()),
                    "admit_dayofweek": int(entries["admit_dayofweek"].get()),
                    "icu_hours": 0,
                    "admission_type": "Manual Entry",
                    "diagnosis": "Not provided",
                    "alert_status": "New",
                    "clinical_note": "",
                    "room_number": f"ICU-{(len(patients) % 15) + 101}"
                }

                result = call_model(patient)
                patients.append(patient)
                save_patient_to_database(patient)
                self.selected_patient = patient

                messagebox.showinfo(
                    "Patient Added",
                    f"Patient {patient['id']} was added successfully.\n\n"
                    f"Risk level: {result['risk_level']}\nRisk: {result['risk_percent']}"
                )
                self.show_patient_summary()

            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Could not add patient:\n{e}")

        ctk.CTkButton(self.main, text="Add Patient & Calculate Risk", width=280, height=45, command=save_patient).pack(pady=25)

    def optional_float(self, value):
        value = str(value).strip()
        if value == "" or value.lower() == "nan":
            return None
        return float(value)

    def upload_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select Patient CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        progress_window = ctk.CTkToplevel(self)
        progress_window.title("Importing Patients")
        progress_window.geometry("500x270")
        progress_window.resizable(False, False)
        progress_window.transient(self)

        ctk.CTkLabel(progress_window, text="Importing Patient Data", font=("Arial", 22, "bold")).pack(pady=(30, 10))
        status_label = ctk.CTkLabel(progress_window, text="Reading CSV file...", font=("Arial", 15))
        status_label.pack(pady=10)

        progress_bar = ctk.CTkProgressBar(progress_window, width=400)
        progress_bar.pack(pady=15)
        progress_bar.set(0)

        percent_label = ctk.CTkLabel(progress_window, text="0%", font=("Arial", 15, "bold"))
        percent_label.pack()

        eta_label = ctk.CTkLabel(progress_window, text="Estimated time remaining: calculating...", font=("Arial", 13))
        eta_label.pack(pady=8)

        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)

        def update_progress(processed, total, start_time):
            if total == 0:
                return
            fraction = processed / total
            percentage = fraction * 100
            elapsed = time.time() - start_time
            remaining = (elapsed / processed) * (total - processed) if processed > 0 else 0

            minutes = int(remaining // 60)
            seconds = int(remaining % 60)

            progress_bar.set(fraction)
            percent_label.configure(text=f"{percentage:.1f}%")
            status_label.configure(text=f"Processing patient {processed:,} of {total:,}")
            eta_label.configure(text=f"Estimated time remaining: {minutes:02d}:{seconds:02d}")

        def import_worker():
            try:
                status_label.after(0, lambda: status_label.configure(text="Reading CSV file..."))
                df = pd.read_csv(file_path)
                df["INTIME"] = pd.to_datetime(df["INTIME"], format="mixed", dayfirst=False, errors="coerce")

                required_columns = [
                    "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "GENDER",
                    "DiasBP_mean", "HeartRate_mean", "SysBP_mean",
                    "DiasBP_min", "HeartRate_min", "SysBP_min",
                    "DiasBP_max", "HeartRate_max", "SysBP_max", "AGE"
                ]

                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    self.after(0, lambda: messagebox.showerror("Invalid CSV", "Missing columns:\n\n" + "\n".join(missing_columns)))
                    self.after(0, progress_window.destroy)
                    return

                total = len(df)
                start_time = time.time()
                all_new_patients = []
                skipped = 0
                BATCH_SIZE = 500

                for batch_start in range(0, total, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, total)
                    df_batch = df.iloc[batch_start:batch_end]
                    batch_patients = []

                    for idx, row in df_batch.iterrows():
                        try:
                            intime = row["INTIME"]
                            if pd.isna(intime):
                                admit_hour, admit_day, intime_text = 12, 2, ""
                            else:
                                admit_hour, admit_day, intime_text = int(intime.hour), int(intime.dayofweek), str(intime)

                            patient = {
                                "id": str(row["SUBJECT_ID"]),
                                "subject_id": str(row["SUBJECT_ID"]),
                                "hadm_id": str(row["HADM_ID"]),
                                "icustay_id": str(row["ICUSTAY_ID"]),
                                "intime": intime_text,
                                "age": float(row["AGE"]),
                                "gender": str(row["GENDER"]).strip().upper(),
                                "heart_rate_mean": float(row["HeartRate_mean"]),
                                "heart_rate_min": float(row["HeartRate_min"]),
                                "heart_rate_max": float(row["HeartRate_max"]),
                                "systolic_bp_mean": float(row["SysBP_mean"]),
                                "systolic_bp_min": float(row["SysBP_min"]),
                                "systolic_bp_max": float(row["SysBP_max"]),
                                "diastolic_bp_mean": float(row["DiasBP_mean"]),
                                "diastolic_bp_min": float(row["DiasBP_min"]),
                                "diastolic_bp_max": float(row["DiasBP_max"]),
                                "creatinine_max": None if pd.isna(row.get("Creatinine_max")) else float(row["Creatinine_max"]),
                                "lactate_max": None if pd.isna(row.get("Lactate_max")) else float(row["Lactate_max"]),
                                "admit_hour": admit_hour,
                                "admit_dayofweek": admit_day,
                                "icu_hours": "N/A",
                                "admission_type": "CSV Import",
                                "diagnosis": "Not provided",
                                "alert_status": "New",
                                "clinical_note": "",
                                "room_number": f"ICU-{(idx % 15) + 101}"
                            }

                            if "HOSPITAL_EXPIRE_FLAG" in df.columns:
                                val = row.get("HOSPITAL_EXPIRE_FLAG")
                                patient["hospital_expire_flag"] = None if pd.isna(val) else int(val)

                            batch_patients.append(patient)
                        except Exception as e:
                            skipped += 1

                    if batch_patients:
                        batch_predict_patients(batch_patients)
                        all_new_patients.extend(batch_patients)

                    processed = batch_end
                    self.after(0, update_progress, processed, total, start_time)

                self.after(0, lambda: status_label.configure(text="Saving patients to database..."))
                self.after(0, lambda: progress_bar.set(0.95))
                self.after(0, lambda: percent_label.configure(text="95%"))

                save_patients_bulk(all_new_patients)

                def finish_import():
                    patients.extend(all_new_patients)
                    if all_new_patients:
                        self.selected_patient = all_new_patients[0]

                    progress_bar.set(1)
                    percent_label.configure(text="100%")
                    status_label.configure(text="Import completed!")
                    eta_label.configure(text="Estimated time remaining: 00:00")

                    messagebox.showinfo("CSV Import Complete", f"Successfully imported: {len(all_new_patients):,} patients\n\nSkipped: {skipped:,}")
                    progress_window.destroy()
                    self.show_dashboard()

                self.after(0, finish_import)

            except Exception as e:
                error_message = str(e)
                def show_error():
                    progress_window.destroy()
                    messagebox.showerror("CSV Import Error", f"Could not import CSV:\n\n{error_message}")
                self.after(0, show_error)

        threading.Thread(target=import_worker, daemon=True).start()

    def show_missing_data(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient

        self.show_title(f"Missing Data Recommendations - {patient['id']}")
        missing = []

        if patient.get("lactate_max") is None:
            missing.append("Lactate is missing — consider ordering lactate test if clinically relevant.")
        if patient.get("creatinine_max") is None:
            missing.append("Creatinine is missing — consider reviewing renal function labs.")

        if not missing:
            ctk.CTkLabel(self.main, text="No important missing data detected.", font=("Arial", 18), text_color="#27ae60").pack(pady=30)
        else:
            for item in missing:
                card = ctk.CTkFrame(self.main)
                card.pack(fill="x", padx=25, pady=10)
                ctk.CTkLabel(card, text=f"• {item}", font=("Arial", 16)).pack(anchor="w", padx=20, pady=12)


if __name__ == "__main__":
    login = LoginWindow()
    login.mainloop()

    logged_user = getattr(login, "logged_user", None)

    try:
        login.destroy()
    except Exception:
        pass

    if logged_user is not None:
        app = ICUApp(logged_user)
        app.mainloop()