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
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "icu_system.db"

from backend.app import (
    compute_features,
    generate_alerts,
    get_risk_drivers,
    model,
    scaler,
    FEATURES,
    THRESHOLD
)
import numpy as np

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    default_users = [
        ("doctor1", hash_password("doctor123"), "Dr. Sarah Cohen", "Doctor"),
        ("nurse1", hash_password("nurse123"), "Nurse Daniel Levi", "Nurse"),
        ("admin1", hash_password("admin123"), "System Administrator", "Admin")
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO users (username, password, full_name, role)
        VALUES (?, ?, ?, ?)
    """, default_users)

    connection.commit()
    connection.close()


def authenticate_user(username, password):
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("""
        SELECT username, password, full_name, role
        FROM users
        WHERE username = ?
    """, (username,))

    row = cursor.fetchone()
    connection.close()

    if row is None or row["password"] != hash_password(password):
        return None

    return {
        "username": row["username"],
        "name": row["full_name"],
        "role": row["role"]
    }


def get_all_users():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("SELECT username, full_name, role FROM users ORDER BY role, full_name")
    rows = cursor.fetchall()
    connection.close()

    return [{"username": row["username"], "name": row["full_name"], "role": row["role"]} for row in rows]


def add_user_to_database(username, password, full_name, role):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, password, full_name, role)
            VALUES (?, ?, ?, ?)
        """, (username, hash_password(password), full_name, role))
        connection.commit()
    finally:
        connection.close()


def delete_user_from_database(username):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
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


patients = []


def initialize_application_data():
    global patients
    init_database()
    patients = load_patients_from_database()


def get_reassessment_info(patient):
    level = patient.get("_risk_level", "LOW")
    last_review = patient.get("last_reviewed_at")

    if not last_review:
        if level in ["CRITICAL", "HIGH"]:
            return "⚠️ Urgent: Pending initial review!", "#c0392b"
        return "ℹ️ Pending review", "#e67e22"

    review_intervals = {"CRITICAL": 15, "HIGH": 30, "MODERATE": 60, "LOW": 240}
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
        self.geometry("520x620")
        self.resizable(False, False)
        self.logged_user = None

        container = ctk.CTkFrame(self, width=440, height=540)
        container.pack(expand=True, padx=30, pady=25)

        ctk.CTkLabel(
            container,
            text="🏥 ICU Decision Support",
            font=("Arial", 24, "bold")
        ).pack(pady=(20, 5))

        ctk.CTkLabel(
            container,
            text="Healthcare Professional Login",
            font=("Arial", 15),
            text_color="gray"
        ).pack(pady=(0, 20))

        self.username_entry = ctk.CTkEntry(
            container,
            width=320,
            height=40,
            placeholder_text="Username"
        )
        self.username_entry.pack(pady=8)

        password_frame = ctk.CTkFrame(container, fg_color="transparent")
        password_frame.pack(pady=8)

        self.password_entry = ctk.CTkEntry(
            password_frame,
            width=275,
            height=40,
            placeholder_text="Password",
            show="*"
        )
        self.password_entry.pack(side="left", padx=(0, 5))

        self.show_password_var = False
        self.toggle_pwd_btn = ctk.CTkButton(
            password_frame,
            text="👁️",
            width=40,
            height=40,
            font=("Arial", 14),
            fg_color="#7f8c8d",
            hover_color="#95a5a6",
            command=self.toggle_password_visibility
        )
        self.toggle_pwd_btn.pack(side="left")

        ctk.CTkButton(
            container,
            text="Login",
            width=320,
            height=40,
            font=("Arial", 14, "bold"),
            command=self.login
        ).pack(pady=15)

        self.status_label = ctk.CTkLabel(
            container,
            text="",
            text_color="red",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=(0, 10))

        self.password_entry.bind(
            "<Return>",
            lambda event: self.login()
        )

        demo_frame = ctk.CTkFrame(container, fg_color="#eef2f7", corner_radius=8)
        demo_frame.pack(fill="x", padx=20, pady=(5, 15))

        ctk.CTkLabel(
            demo_frame,
            text="🛠️ Demo Quick Fill (Presentation Mode Only)",
            font=("Arial", 11, "bold"),
            text_color="#576574"
        ).pack(pady=(8, 4))

        quick_buttons_frame = ctk.CTkFrame(demo_frame, fg_color="transparent")
        quick_buttons_frame.pack(pady=(0, 8))

        ctk.CTkButton(
            quick_buttons_frame,
            text="🩺 Doctor",
            width=85,
            height=28,
            font=("Arial", 11),
            fg_color="#34495e",
            hover_color="#2c3e50",
            command=lambda: self.fill_credentials("doctor1", "doctor123")
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            quick_buttons_frame,
            text="💉 Nurse",
            width=85,
            height=28,
            font=("Arial", 11),
            fg_color="#34495e",
            hover_color="#2c3e50",
            command=lambda: self.fill_credentials("nurse1", "nurse123")
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            quick_buttons_frame,
            text="⚙️ Admin",
            width=85,
            height=28,
            font=("Arial", 11),
            fg_color="#34495e",
            hover_color="#2c3e50",
            command=lambda: self.fill_credentials("admin1", "admin123")
        ).pack(side="left", padx=4)

    def toggle_password_visibility(self):
        if self.show_password_var:
            self.password_entry.configure(show="*")
            self.toggle_pwd_btn.configure(text="👁️", fg_color="#7f8c8d")
            self.show_password_var = False
        else:
            self.password_entry.configure(show="")
            self.toggle_pwd_btn.configure(text="🙈", fg_color="#2980b9")
            self.show_password_var = True

    def fill_credentials(self, username, password):
        self.username_entry.delete(0, "end")
        self.username_entry.insert(0, username)
        self.password_entry.delete(0, "end")
        self.password_entry.insert(0, password)
        self.status_label.configure(text="")

    def login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get()

        user = authenticate_user(username, password)

        if user is None:
            self.status_label.configure(
                text="Invalid username or password."
            )
            return

        self.logged_user = user
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
        self.switch_user_requested = False

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

    def switch_user(self):
        confirm = messagebox.askyesno(
            "Switch User",
            "Do you want to sign out and switch to another user?"
        )

        if not confirm:
            return

        self.switch_user_requested = True
        self.quit()

    def toggle_appearance_mode(self):
        current_mode = ctk.get_appearance_mode()
        if current_mode == "Light":
            ctk.set_appearance_mode("Dark")
            self.theme_button.configure(text="☀️  Light Mode")
        else:
            ctk.set_appearance_mode("Light")
            self.theme_button.configure(text="🌙  Dark Mode")

    def check_overdue_critical_patients(self):
        overdue_critical_patients = []
        for p in patients:
            if get_missing_model_fields(p):
                continue
            _, level = calculate_risk_score(p)
            if level == "CRITICAL":
                info_text, _ = get_reassessment_info(p)
                if "Due for re-assessment" in info_text or "Pending initial review" in info_text:
                    overdue_critical_patients.append(p)

        if overdue_critical_patients:
            banner = ctk.CTkFrame(self.main, fg_color="#c0392b", corner_radius=8)
            banner.pack(fill="x", padx=25, pady=(0, 15))

            count = len(overdue_critical_patients)
            msg = f"🚨 URGENT CLINICAL ATTENTION: {count} CRITICAL patient{'s require' if count > 1 else ' requires'} immediate vital re-assessment!"

            ctk.CTkLabel(
                banner,
                text=msg,
                font=("Arial", 14, "bold"),
                text_color="white"
            ).pack(side="left", padx=20, pady=10)

            def go_to_priority_critical():
                self.priority_filter = "CRITICAL"
                self.priority_page = 0
                self.set_active_button("Priority Queue")
                self.show_priority_queue()

            ctk.CTkButton(
                banner,
                text="View in Priority Queue ➔",
                font=("Arial", 12, "bold"),
                fg_color="white",
                text_color="#c0392b",
                hover_color="#f2f2f2",
                height=30,
                command=go_to_priority_critical
            ).pack(side="right", padx=15, pady=8)

    def show_admin_console(self):
        if self.logged_user.get("role") != "Admin":
            messagebox.showerror(
                "Access Denied",
                "Only administrators can access the Admin Console."
            )
            return

        self.clear_main()
        self.show_title("Admin Console")

        add_frame = ctk.CTkFrame(self.main)
        add_frame.pack(fill="x", padx=30, pady=(5, 20))

        ctk.CTkLabel(add_frame, text="Add New User", font=("Arial", 20, "bold")).pack(anchor="w", padx=20, pady=(15, 10))

        form = ctk.CTkFrame(add_frame, fg_color="transparent")
        form.pack(fill="x", padx=20, pady=10)

        username_entry = ctk.CTkEntry(form, width=200, placeholder_text="Username")
        username_entry.grid(row=0, column=0, padx=8, pady=8)

        name_entry = ctk.CTkEntry(form, width=220, placeholder_text="Full Name")
        name_entry.grid(row=0, column=1, padx=8, pady=8)

        password_entry = ctk.CTkEntry(form, width=200, placeholder_text="Password", show="*" )
        password_entry.grid(row=0, column=2, padx=8, pady=8)

        role_var = ctk.StringVar(value="Nurse")
        role_menu = ctk.CTkOptionMenu(form, values=["Doctor", "Nurse", "Admin"], variable=role_var, width=140)
        role_menu.grid(row=0, column=3, padx=8, pady=8)

        def add_new_user():
            username = username_entry.get().strip()
            full_name = name_entry.get().strip()
            password = password_entry.get()
            role = role_var.get()

            if not username or not full_name or not password:
                messagebox.showwarning("Missing Information", "All fields are required.")
                return

            if len(password) < 6:
                messagebox.showwarning("Invalid Password", "Password must contain at least 6 characters.")
                return

            try:
                add_user_to_database(username, password, full_name, role)
                messagebox.showinfo("User Added", f"User '{username}' was created successfully.")
                self.show_admin_console()
            except sqlite3.IntegrityError:
                messagebox.showerror("User Exists", f"The username '{username}' already exists.")

        ctk.CTkButton(add_frame, text="➕ Add User", width=150, command=add_new_user).pack(anchor="e", padx=25, pady=(5, 15))

        users_frame = ctk.CTkFrame(self.main)
        users_frame.pack(fill="x", padx=30, pady=10)

        ctk.CTkLabel(users_frame, text="System Users", font=("Arial", 20, "bold")).pack(anchor="w", padx=20, pady=(15, 10))

        users = get_all_users()
        table = ctk.CTkFrame(users_frame)
        table.pack(fill="x", padx=20, pady=(0, 20))

        headers = ["Username", "Name", "Role", "Action"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(table, text=header, font=("Arial", 14, "bold")).grid(row=0, column=col, padx=20, pady=10)

        for row_number, user in enumerate(users, start=1):
            ctk.CTkLabel(table, text=user["username"]).grid(row=row_number, column=0, padx=20, pady=8)
            ctk.CTkLabel(table, text=user["name"]).grid(row=row_number, column=1, padx=20, pady=8)
            ctk.CTkLabel(table, text=user["role"]).grid(row=row_number, column=2, padx=20, pady=8)

            def delete_selected_user(username=user["username"], name=user["name"]):
                if username == self.logged_user["username"]:
                    messagebox.showwarning("Cannot Delete User", "You cannot delete the account you are currently logged in with.")
                    return

                confirm = messagebox.askyesno("Delete User", f"Are you sure you want to delete {name} ({username})?")
                if confirm:
                    delete_user_from_database(username)
                    messagebox.showinfo("User Deleted", f"User '{username}' was deleted.")
                    self.show_admin_console()

            ctk.CTkButton(
                table, text="🗑 Delete", width=90, fg_color="#c0392b", hover_color="#922b21",
                command=delete_selected_user
            ).grid(row=row_number, column=3, padx=20, pady=8)

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
                    ("Department KPI", "📈  Department KPI", self.show_department_kpi),
                    ("What-if Analysis", "🧪  What-if Analysis", self.show_what_if),
                    ("Trend Monitoring", "📉  Trend Monitoring", self.show_trends),
                    ("Missing Data", "⚠️  Missing Data", self.show_missing_data),
                ]
            ),
        ]

        if self.logged_user.get("role") == "Admin":
            sections.append(
                (
                    "ADMINISTRATION",
                    [
                        ("Admin Console", "⚙️  Admin Console", self.show_admin_console)
                    ]
                )
            )

        self.sidebar_buttons = {}

        for section_title, buttons in sections:
            ctk.CTkLabel(
                self.sidebar,
                text=section_title,
                font=("Arial", 10, "bold"),
                text_color="#7f8c8d",
                anchor="w"
            ).pack(fill="x", padx=16, pady=(6, 2))

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

        ctk.CTkFrame(
            self.sidebar,
            height=2,
            fg_color="#d0d0d0"
        ).pack(fill="x", padx=15, pady=(10, 4))

        current_icon = "🌙  Dark Mode" if ctk.get_appearance_mode() == "Light" else "☀️  Light Mode"
        self.theme_button = ctk.CTkButton(
            self.sidebar,
            text=current_icon,
            anchor="w",
            height=30,
            font=("Arial", 12),
            fg_color="#4a6572",
            hover_color="#34495e",
            command=self.toggle_appearance_mode
        )
        self.theme_button.pack(fill="x", padx=12, pady=(2, 4))

        ctk.CTkButton(
            self.sidebar,
            text="🔄  Switch User",
            anchor="w",
            height=30,
            font=("Arial", 12),
            fg_color="#6c757d",
            hover_color="#5a6268",
            command=self.switch_user
        ).pack(fill="x", padx=12, pady=(2, 10))

    def edit_patient_room(self, patient):
        user_role = getattr(self, "logged_user", {}).get("role", "")
        if user_role != "Doctor":
            messagebox.showwarning("Permission Denied", "Only Doctors are authorized to update patient room assignments.")
            return

        room_win = ctk.CTkToplevel(self)
        room_win.title("Assign / Edit Room Number")
        room_win.geometry("380x220")
        room_win.resizable(False, False)
        room_win.transient(self)

        ctk.CTkLabel(room_win, text=f"Edit Room for Patient {patient.get('id')}", font=("Arial", 16, "bold")).pack(pady=(20, 10))

        input_frame = ctk.CTkFrame(room_win, fg_color="transparent")
        input_frame.pack(pady=10)

        ctk.CTkLabel(input_frame, text="Room: ICU-", font=("Arial", 16, "bold")).pack(side="left")

        current_room = str(patient.get("room_number", "")).replace("ICU-", "").strip()

        room_entry = ctk.CTkEntry(input_frame, width=120, placeholder_text="e.g. 104")
        room_entry.pack(side="left", padx=5)
        if current_room:
            room_entry.insert(0, current_room)

        def save_room_number():
            raw_digits = room_entry.get().strip()

            if not raw_digits:
                messagebox.showerror("Invalid Input", "Room number cannot be empty.")
                return

            if not raw_digits.isdigit():
                messagebox.showerror("Invalid Input", "Room identifier must contain digits only (e.g. 104).")
                return

            room_int = int(raw_digits)
            if room_int < 1 or room_int > 999:
                messagebox.showerror("Invalid Input", "Room number must be between 1 and 999.")
                return

            full_room_name = f"ICU-{raw_digits}"
            patient["room_number"] = full_room_name
            self.selected_patient = patient

            for p in patients:
                if str(p.get("subject_id", p.get("id"))) == str(patient.get("subject_id", patient.get("id"))):
                    p["room_number"] = full_room_name
                    break

            save_patient_to_database(patient)

            if hasattr(self, "room_label_var") and self.room_label_var:
                self.room_label_var.configure(text=f"Room Assignment: {full_room_name}")
            else:
                self.show_patient_summary()

            room_win.destroy()
            messagebox.showinfo("Room Updated", f"Patient {patient.get('id')} room successfully assigned to {full_room_name}.")

        ctk.CTkButton(room_win, text="Save Room", width=140, command=save_room_number).pack(pady=15)

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
        ctk.CTkLabel(self.main, text=text, font=("Arial", 28, "bold")).pack(pady=20)

    def select_patient(self, patient):
        self.selected_patient = patient
        self.set_active_button("Patient Summary")
        self.show_patient_summary()

    def show_dashboard(self):
        self.clear_main()
        self.show_title("ICU Dashboard")
        self.check_overdue_critical_patients()

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
            search_frame, values=["Patient ID", "Hospital Admission ID", "ICU Stay ID"],
            variable=search_field_var, width=190
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

        ctk.CTkButton(search_frame, text="Search 🔍", width=90, command=apply_search).pack(side="left", padx=5)
        ctk.CTkButton(search_frame, text="Clear 🧹", width=80, command=clear_search).pack(side="left", padx=5)
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

            ctk.CTkButton(frame, text="Open 👤", width=80, command=lambda p=patient: self.select_patient(p)).grid(row=row, column=7, padx=12, pady=8)

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

        ctk.CTkButton(filter_frame, text="Apply 🔍", width=90, command=apply_filter).pack(side="left", padx=10)

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

            header_frame = ctk.CTkFrame(card, fg_color="transparent")
            header_frame.pack(fill="x", padx=20, pady=(10, 2))

            ctk.CTkLabel(
                header_frame,
                text=f"{index}. Patient {patient_id} ({room_num}) — {level} risk — {score}/100",
                font=("Arial", 19, "bold"),
                text_color=get_risk_color(level)
            ).pack(side="left")

            reassess_text, reassess_color = get_reassessment_info(patient)
            ctk.CTkLabel(
                header_frame,
                text=reassess_text,
                font=("Arial", 13, "bold"),
                text_color=reassess_color
            ).pack(side="right")

            actions = get_recommended_actions(level)
            if actions:
                ctk.CTkLabel(card, text=f"Suggested priority: {actions[0]}", font=("Arial", 14)).pack(anchor="w", padx=20, pady=4)

            bottom_bar = ctk.CTkFrame(card, fg_color="transparent")
            bottom_bar.pack(fill="x", padx=20, pady=(4, 10))

            last_review = patient.get("last_reviewed_at", "Never")
            reviewer = patient.get("reviewed_by", "N/A")
            ctk.CTkLabel(
                bottom_bar,
                text=f"⏱️ Last Reviewed: {last_review} ({reviewer})",
                font=("Arial", 12),
                text_color="gray"
            ).pack(side="left")

            action_buttons = ctk.CTkFrame(bottom_bar, fg_color="transparent")
            action_buttons.pack(side="right")

            ctk.CTkButton(
                action_buttons, text="Mark Reviewed ✅", width=110, height=26, font=("Arial", 11),
                command=lambda p=patient: self.quick_review_patient(p, "Reviewed")
            ).pack(side="left", padx=3)

            ctk.CTkButton(
                action_buttons, text="Open Patient 👤", width=100, height=26, font=("Arial", 11, "bold"),
                command=lambda p=patient: self.select_patient(p)
            ).pack(side="left", padx=3)

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

    def quick_review_patient(self, patient, status):
        patient["alert_status"] = status
        patient["last_reviewed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        patient["reviewed_by"] = getattr(self, "logged_user", {}).get("name", "Staff") if hasattr(self, "logged_user") else "Staff"

        save_patient_to_database(patient)
        messagebox.showinfo("Patient Reviewed", f"Patient {patient.get('id')} review recorded. Timer reset!")
        self.show_priority_queue()

    def show_patient_summary(self):
        """מסך סיכום מטופל בעיצוב מאוזן ונקי עם סרגל פעולות בתחתית"""
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient
        score, level = calculate_risk_score(patient)
        result = patient.get("model_result", {})

        self.show_title(f"Patient Summary Card - {patient['id']}")

        card = ctk.CTkFrame(self.main)
        card.pack(fill="x", padx=25, pady=15)

        # 1. שורת חדר עליונה
        room_num = patient.get("room_number", "Unassigned")
        room_frame = ctk.CTkFrame(card, fg_color="transparent")
        room_frame.pack(anchor="w", padx=25, pady=(15, 8))

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
            ctk.CTkLabel(card, text=line, font=("Arial", 17)).pack(anchor="w", padx=25, pady=4)

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

        # 4. המלצות קליניות
        ctk.CTkLabel(card, text="Suggested next steps:", font=("Arial", 18, "bold")).pack(anchor="w", padx=25, pady=(15, 6))

        for action in get_recommended_actions(level):
            ctk.CTkLabel(card, text=f"• {action}", font=("Arial", 15)).pack(anchor="w", padx=45, pady=3)

        # ----------------------------------------------------
        # 5. סרגל פעולות בתחתית הכרטיס (Bottom Action Bar)
        # ----------------------------------------------------
        summary_actions_bar = ctk.CTkFrame(card, fg_color="transparent")
        summary_actions_bar.pack(fill="x", padx=25, pady=(20, 15))

        ctk.CTkButton(
            summary_actions_bar,
            text="📥  Export Medical Report (.txt)",
            width=220,
            height=36,
            font=("Arial", 13, "bold"),
            fg_color="#27ae60",
            hover_color="#1e8449",
            command=lambda: self.export_patient_report(patient)
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            summary_actions_bar,
            text="✏️  Quick Edit Patient Vitals & Labs",
            width=230,
            height=36,
            font=("Arial", 13),
            fg_color="#2980b9",
            hover_color="#1f618d",
            command=lambda: self.edit_patient_vitals(patient)
        ).pack(side="right")

    def export_patient_report(self, patient):
        if not patient:
            return

        patient_id = patient.get("subject_id", patient.get("id", "Unknown"))
        room_num = patient.get("room_number", "Unassigned")
        score, level = calculate_risk_score(patient)
        result = patient.get("model_result", {})
        doctor_note = patient.get("clinical_note", "") or "None provided"
        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_content = f"""================================================================================
                    ICU CLINICAL DECISION SUPPORT SYSTEM
                        PATIENT SUMMARY & RISK REPORT
================================================================================
Generated On: {generated_at}
Generated By: {getattr(self, 'logged_user', {}).get('name', 'Staff')} ({getattr(self, 'logged_user', {}).get('role', 'User')})

--------------------------------------------------------------------------------
1. PATIENT DEMOGRAPHICS & LOCATION
--------------------------------------------------------------------------------
Patient ID / Subject ID: {patient_id}
Hospital Admission ID:   {patient.get('hadm_id', 'N/A')}
ICU Stay ID:             {patient.get('icustay_id', 'N/A')}
Room Assignment:         {room_num}
Age:                     {patient.get('age', 'N/A')}
Gender:                  {patient.get('gender', 'N/A')}
Admission Type:          {patient.get('admission_type', 'N/A')}
Diagnosis:               {patient.get('diagnosis', 'N/A')}

--------------------------------------------------------------------------------
2. ML RISK ASSESSMENT & TRIAGE
--------------------------------------------------------------------------------
Risk Level:              {level}
Risk Score:              {score} / 100
Model Prediction:        {result.get('prediction', 'N/A')}
Alert Status:            {patient.get('alert_status', 'New')}

--------------------------------------------------------------------------------
3. VITAL SIGNS & LAB VALUES SUMMARY
--------------------------------------------------------------------------------
Heart Rate (Min / Mean / Max):       {patient.get('heart_rate_min')} / {patient.get('heart_rate_mean')} / {patient.get('heart_rate_max')} bpm
Systolic BP (Min / Mean / Max):      {patient.get('systolic_bp_min')} / {patient.get('systolic_bp_mean')} / {patient.get('systolic_bp_max')} mmHg
Diastolic BP (Min / Mean / Max):     {patient.get('diastolic_bp_min')} / {patient.get('diastolic_bp_mean')} / {patient.get('diastolic_bp_max')} mmHg
Creatinine Max:                      {patient.get('creatinine_max', 'Missing')}
Lactate Max:                         {patient.get('lactate_max', 'Missing')}

--------------------------------------------------------------------------------
4. PHYSICIAN OVERRIDE & CLINICAL JUSTIFICATION
--------------------------------------------------------------------------------
Physician Note:
{doctor_note}

--------------------------------------------------------------------------------
5. RECOMMENDED CLINICAL ACTIONS
--------------------------------------------------------------------------------
"""
        for action in get_recommended_actions(level):
            report_content += f"• {action}\n"

        report_content += """
================================================================================
CONFIDENTIAL MEDICAL RECORD - FOR AUTHORIZED CLINICAL USE ONLY
================================================================================
"""

        default_filename = f"Patient_{patient_id}_ICU_Report.txt"
        file_path = filedialog.asksaveasfilename(
            title="Select Path to Save Medical Report",
            initialfile=default_filename,
            defaultextension=".txt",
            filetypes=[("Text Documents", "*.txt"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
                messagebox.showinfo("Report Exported", f"Medical report successfully saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not save report file:\n{e}")

    def show_patient_details(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient
        self.show_title(f"Patient Details - {patient['id']}")

        main_layout_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        main_layout_frame.pack(fill="x", padx=25, pady=10)

        details_frame = ctk.CTkFrame(main_layout_frame)
        details_frame.pack(side="left", fill="both", expand=True, padx=(0, 20))

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

        for key, value in details:
            ctk.CTkLabel(details_frame, text=f"{key}: {value}", font=("Arial", 16)).pack(anchor="w", padx=25, pady=5)

        action_frame = ctk.CTkFrame(main_layout_frame, fg_color="transparent")
        action_frame.pack(side="right", anchor="n", padx=10)

        ctk.CTkButton(
            action_frame,
            text="✏️ Edit Patient Vitals & Labs",
            width=220,
            height=36,
            font=("Arial", 13, "bold"),
            command=lambda: self.edit_patient_vitals(patient)
        ).pack(pady=(0, 20))

        gender = str(patient.get("gender", "M")).strip().upper()
        image_name = "female.png" if gender == "F" else "male.png"

        possible_paths = [
            APP_DIR / image_name,
            APP_DIR / "assets" / image_name,
            APP_DIR / "images" / image_name
        ]

        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                break

        if image_path:
            try:
                pil_img = Image.open(image_path)
                gender_ctk_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(140, 160))
                img_label = ctk.CTkLabel(action_frame, image=gender_ctk_image, text="")
                img_label.pack(pady=10)
            except Exception as e:
                print(f"Could not load gender image: {e}")

    def edit_patient_vitals(self, patient):
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
                parsed_values = {}

                for key, label_text, _ in fields_to_edit:
                    raw_val = entries[key].get().strip()

                    if key in ["lactate_max", "creatinine_max"]:
                        if raw_val == "" or raw_val.lower() in ["none", "nan"]:
                            parsed_values[key] = None
                            continue

                    if not raw_val:
                        raise ValueError(f"'{label_text}' is required.")

                    try:
                        v = float(raw_val)
                        parsed_values[key] = v
                    except ValueError:
                        raise ValueError(f"'{label_text}' must be a valid numeric value.")

                age = parsed_values["age"]
                if age < 0 or age > 120:
                    raise ValueError("Age must be between 0 and 120.")

                positive_vitals = [
                    ("heart_rate_mean", "Heart Rate Mean"),
                    ("heart_rate_min", "Heart Rate Min"),
                    ("heart_rate_max", "Heart Rate Max"),
                    ("systolic_bp_mean", "Systolic BP Mean"),
                    ("systolic_bp_min", "Systolic BP Min"),
                    ("systolic_bp_max", "Systolic BP Max"),
                    ("diastolic_bp_mean", "Diastolic BP Mean"),
                    ("diastolic_bp_min", "Diastolic BP Min"),
                    ("diastolic_bp_max", "Diastolic BP Max"),
                ]

                for key, label in positive_vitals:
                    if parsed_values[key] <= 0:
                        raise ValueError(f"'{label}' must be greater than 0.")

                if parsed_values["lactate_max"] is not None and parsed_values["lactate_max"] < 0:
                    raise ValueError("Lactate Max cannot be negative.")

                if parsed_values["creatinine_max"] is not None and parsed_values["creatinine_max"] < 0:
                    raise ValueError("Creatinine Max cannot be negative.")

                if not (parsed_values["heart_rate_min"] <= parsed_values["heart_rate_mean"] <= parsed_values["heart_rate_max"]):
                    raise ValueError("Heart Rate values must satisfy: Min ≤ Mean ≤ Max.")

                if not (parsed_values["systolic_bp_min"] <= parsed_values["systolic_bp_mean"] <= parsed_values["systolic_bp_max"]):
                    raise ValueError("Systolic BP values must satisfy: Min ≤ Mean ≤ Max.")

                if not (parsed_values["diastolic_bp_min"] <= parsed_values["diastolic_bp_mean"] <= parsed_values["diastolic_bp_max"]):
                    raise ValueError("Diastolic BP values must satisfy: Min ≤ Mean ≤ Max.")

                for key, val in parsed_values.items():
                    patient[key] = val

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

            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
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
        self.show_title(f"Why This Risk Score? - Patient {patient['id']}")

        result = patient.get("model_result")
        if not result or not result.get("risk_drivers"):
            missing_fields = get_missing_model_fields(patient)
            if missing_fields:
                card = ctk.CTkFrame(self.main)
                card.pack(fill="x", padx=25, pady=20)
                ctk.CTkLabel(card, text="⚠️ Risk Explanation Unavailable", font=("Arial", 18, "bold"),
                             text_color="#c0392b").pack(anchor="w", padx=20, pady=(15, 5))
                ctk.CTkLabel(card,
                             text="Core physiological data is missing to explain the ML model decision. Missing fields:",
                             font=("Arial", 14)).pack(anchor="w", padx=20, pady=(0, 10))
                for field in missing_fields:
                    ctk.CTkLabel(card, text=f"• {field.replace('_', ' ').title()}", font=("Arial", 14, "bold")).pack(
                        anchor="w", padx=35, pady=2)

                ctk.CTkButton(
                    self.main,
                    text="✏️  Enter Missing Vitals Now",
                    font=("Arial", 13, "bold"),
                    height=36,
                    command=lambda: self.edit_patient_vitals(patient)
                ).pack(pady=15)
                return

            try:
                result = call_model(patient)
            except Exception as e:
                ctk.CTkLabel(self.main, text=f"Could not generate risk explanation:\n{e}", font=("Arial", 16),
                             text_color="#c0392b").pack(pady=25)
                return

        drivers = result.get("risk_drivers", [])
        if not drivers:
            ctk.CTkLabel(self.main, text="No significant risk drivers were identified for this patient.",
                         font=("Arial", 17)).pack(pady=25)
            return

        ctk.CTkLabel(self.main, text="Top Physiological Factors Influencing ML Risk Score:",
                     font=("Arial", 18, "bold")).pack(anchor="w", padx=30, pady=(5, 12))

        for driver in drivers:
            direction = driver.get("direction")
            arrow = "▲ Increases Risk of Deterioration" if direction == "increase" else "▼ Decreases Risk (Protective Factor)"
            color = "#c0392b" if direction == "increase" else "#27ae60"

            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=8)

            header_f = ctk.CTkFrame(card, fg_color="transparent")
            header_f.pack(fill="x", padx=20, pady=(10, 3))

            ctk.CTkLabel(header_f, text=f"{driver.get('name', 'Unknown factor')} — {driver.get('value', 'N/A')}",
                         font=("Arial", 17, "bold")).pack(side="left")
            ctk.CTkLabel(header_f, text=f"Impact: {driver.get('impact', 'N/A')}", font=("Arial", 13),
                         text_color="gray").pack(side="right")

            ctk.CTkLabel(card, text=arrow, text_color=color, font=("Arial", 14, "bold")).pack(anchor="w", padx=20,
                                                                                              pady=(0, 10))

        actions_bar = ctk.CTkFrame(self.main, fg_color="transparent")
        actions_bar.pack(fill="x", padx=25, pady=(15, 25))

        def copy_drivers_to_note():
            user_role = getattr(self, "logged_user", {}).get("role", "")
            if user_role != "Doctor":
                messagebox.showwarning("Permission Denied",
                                       "Only Doctors are authorized to append clinical justification notes.")
                return

            driver_summary = ", ".join([f"{d.get('name')}: {d.get('value')}" for d in drivers[:3]])
            driver_text = f"[ML RISK DRIVERS: {driver_summary} logged at {datetime.datetime.now().strftime('%H:%M')}]"

            existing_note = patient.get("clinical_note", "")
            updated_note = f"{existing_note}\n{driver_text}".strip() if existing_note else driver_text
            patient["clinical_note"] = updated_note
            self.selected_patient = patient

            for p in patients:
                if str(p.get("subject_id", p.get("id"))) == str(patient.get("subject_id", patient.get("id"))):
                    p["clinical_note"] = updated_note
                    break

            save_patient_to_database(patient)
            messagebox.showinfo("Note Updated",
                                "Key risk drivers were automatically formatted and attached to Physician Note.")

        ctk.CTkButton(
            actions_bar,
            text="🩺  View Recommended Actions for Drivers ➔",
            font=("Arial", 13, "bold"),
            fg_color="#2980b9",
            hover_color="#1f618d",
            height=36,
            command=self.show_recommended_actions
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            actions_bar,
            text="📋  Insert Top Drivers into Physician Note",
            font=("Arial", 13, "bold"),
            fg_color="#34495e",
            hover_color="#2c3e50",
            height=36,
            command=copy_drivers_to_note
        ).pack(side="right")

    def show_recommended_actions(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient
        score, level = calculate_risk_score(patient)

        self.show_title(f"Recommended Actions & Protocol Checklist - Patient {patient['id']}")

        status_card = ctk.CTkFrame(self.main)
        status_card.pack(fill="x", padx=25, pady=(5, 15))

        status_header = ctk.CTkFrame(status_card, fg_color="transparent")
        status_header.pack(fill="x", padx=20, pady=12)

        ctk.CTkLabel(
            status_header,
            text=f"Current Stratified Risk: {level} ({score}/100)",
            font=("Arial", 20, "bold"),
            text_color=get_risk_color(level)
        ).pack(side="left")

        ctk.CTkLabel(
            status_header,
            text=f"Room: {patient.get('room_number', 'Unassigned')}",
            font=("Arial", 16, "bold")
        ).pack(side="right")

        ctk.CTkLabel(
            self.main,
            text="Clinical Protocol Action Checklist (Check items as executed):",
            font=("Arial", 16, "bold")
        ).pack(anchor="w", padx=30, pady=(5, 10))

        actions = get_recommended_actions(level)
        action_checkboxes = []

        checklist_frame = ctk.CTkFrame(self.main)
        checklist_frame.pack(fill="x", padx=25, pady=5)

        for action in actions:
            row = ctk.CTkFrame(checklist_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=8)

            chk_var = ctk.BooleanVar(value=False)
            chk = ctk.CTkCheckBox(
                row,
                text=action,
                variable=chk_var,
                font=("Arial", 15),
                checkbox_width=24,
                checkbox_height=24
            )
            chk.pack(side="left", padx=5)
            action_checkboxes.append((action, chk_var))

        ctk.CTkLabel(
            self.main,
            text="* Note: Automated clinical decision support recommendations do not substitute physician judgment.",
            text_color="gray",
            font=("Arial", 13, "italic")
        ).pack(anchor="w", padx=30, pady=(10, 15))

        actions_bar = ctk.CTkFrame(self.main, fg_color="transparent")
        actions_bar.pack(fill="x", padx=25, pady=(5, 25))

        def complete_selected_actions():
            completed = [act for act, var in action_checkboxes if var.get()]
            if not completed:
                messagebox.showwarning("No Items Selected", "Please check at least one executed clinical action.")
                return

            completed_text = ", ".join(completed)
            staff_name = getattr(self, "logged_user", {}).get("name", "Staff")
            log_entry = f"[PROTOCOL EXECUTED: {completed_text} completed at {datetime.datetime.now().strftime('%H:%M')} by {staff_name}]"

            existing_note = patient.get("clinical_note", "")
            updated_note = f"{existing_note}\n{log_entry}".strip() if existing_note else log_entry
            patient["clinical_note"] = updated_note
            patient["alert_status"] = "In Progress"
            patient["last_reviewed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            patient["reviewed_by"] = staff_name
            self.selected_patient = patient

            for p in patients:
                if str(p.get("subject_id", p.get("id"))) == str(patient.get("subject_id", patient.get("id"))):
                    p.update(patient)
                    break

            save_patient_to_database(patient)
            messagebox.showinfo("Protocol Acknowledged",
                                f"{len(completed)} action(s) marked as completed and appended to patient record.")
            self.show_recommended_actions()

        def add_clinical_override():
            user_role = getattr(self, "logged_user", {}).get("role", "")
            if user_role != "Doctor":
                messagebox.showwarning("Permission Denied", "Only Doctors are authorized to log protocol overrides.")
                return
            self.edit_clinical_note(patient)

        ctk.CTkButton(
            actions_bar,
            text="✅  Acknowledge & Save Executed Protocol",
            font=("Arial", 13, "bold"),
            fg_color="#27ae60",
            hover_color="#1e8449",
            height=36,
            command=complete_selected_actions
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            actions_bar,
            text="📝  Clinical Override / Reason for Deviation",
            font=("Arial", 13),
            fg_color="#34495e",
            hover_color="#2c3e50",
            height=36,
            command=add_clinical_override
        ).pack(side="right")

    def show_department_kpi(self):
        self.clear_main()
        self.show_title("ICU Department Analytics & KPI")

        if not patients:
            ctk.CTkLabel(
                self.main,
                text="No patient data available to generate department analytics.",
                font=("Arial", 16)
            ).pack(pady=40)
            return

        total_patients = len(patients)

        risk_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0, "INCOMPLETE": 0, "ERROR": 0}
        active_alerts_count = 0
        occupied_rooms = set()
        total_hr, total_sys_bp, total_dias_bp, total_age = 0, 0, 0, 0
        valid_vitals_count = 0

        for p in patients:
            score, level = calculate_risk_score(p)
            risk_counts[level] = risk_counts.get(level, 0) + 1

            if p.get("alert_status", "New") not in ["Resolved", "Inactive"]:
                active_alerts_count += 1

            if p.get("room_number"):
                occupied_rooms.add(p["room_number"])

            if p.get("heart_rate_mean") and p.get("systolic_bp_mean") and p.get("diastolic_bp_mean") and p.get("age"):
                try:
                    total_hr += float(p["heart_rate_mean"])
                    total_sys_bp += float(p["systolic_bp_mean"])
                    total_dias_bp += float(p["diastolic_bp_mean"])
                    total_age += float(p["age"])
                    valid_vitals_count += 1
                except (ValueError, TypeError):
                    pass

        kpi_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        kpi_frame.pack(fill="x", padx=25, pady=(5, 20))

        kpis = [
            ("Total Patients", f"{total_patients:,}", "#2980b9"),
            ("Critical Risk", f"{risk_counts['CRITICAL']:,}", "#c0392b"),
            ("Active Alerts", f"{active_alerts_count:,}", "#e67e22"),
            ("Beds Occupied", f"{len(occupied_rooms):,}", "#27ae60")
        ]

        for i, (title, val, color) in enumerate(kpis):
            card = ctk.CTkFrame(kpi_frame, height=100)
            card.pack(side="left", fill="both", expand=True, padx=8)
            ctk.CTkLabel(card, text=title, font=("Arial", 14), text_color="gray").pack(pady=(15, 2))
            ctk.CTkLabel(card, text=val, font=("Arial", 26, "bold"), text_color=color).pack(pady=(0, 15))

        risk_frame = ctk.CTkFrame(self.main)
        risk_frame.pack(fill="x", padx=25, pady=10)

        ctk.CTkLabel(risk_frame, text="Department Risk Stratification", font=("Arial", 18, "bold")).pack(anchor="w", padx=20, pady=(15, 10))

        risk_levels_meta = [
            ("CRITICAL", "Immediate physician evaluation required", "#c0392b"),
            ("HIGH", "Close telemetry and vital monitoring", "#e67e22"),
            ("MODERATE", "Periodic assessment and lab review", "#f39c12"),
            ("LOW", "Standard routine clinical care", "#27ae60"),
        ]

        for level, desc, color in risk_levels_meta:
            count = risk_counts.get(level, 0)
            pct = (count / total_patients * 100) if total_patients > 0 else 0
            row = ctk.CTkFrame(risk_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=6)

            ctk.CTkLabel(row, text=level, width=110, anchor="w", font=("Arial", 15, "bold"), text_color=color).pack(side="left")
            ctk.CTkLabel(row, text=f"{count:,} patients ({pct:.1f}%)", width=170, anchor="w", font=("Arial", 14, "bold")).pack(side="left")
            ctk.CTkLabel(row, text=f"• {desc}", anchor="w", font=("Arial", 13), text_color="gray").pack(side="left", padx=10)

        ctk.CTkFrame(risk_frame, height=10, fg_color="transparent").pack()

        if valid_vitals_count > 0:
            avg_hr = total_hr / valid_vitals_count
            avg_sys = total_sys_bp / valid_vitals_count
            avg_dias = total_dias_bp / valid_vitals_count
            avg_age = total_age / valid_vitals_count

            vitals_frame = ctk.CTkFrame(self.main)
            vitals_frame.pack(fill="x", padx=25, pady=15)

            ctk.CTkLabel(vitals_frame, text="Department Clinical Averages", font=("Arial", 18, "bold")).pack(anchor="w", padx=20, pady=(15, 10))

            avg_grid = ctk.CTkFrame(vitals_frame, fg_color="transparent")
            avg_grid.pack(fill="x", padx=20, pady=(0, 15))

            avg_stats = [
                ("Average Patient Age", f"{avg_age:.1f} yrs"),
                ("Average Heart Rate", f"{avg_hr:.1f} bpm"),
                ("Average Blood Pressure", f"{avg_sys:.1f} / {avg_dias:.1f} mmHg"),
                ("Monitored Patients", f"{valid_vitals_count:,}")
            ]

            for idx, (stat_title, stat_val) in enumerate(avg_stats):
                r, c = divmod(idx, 2)
                stat_box = ctk.CTkFrame(avg_grid, fg_color="transparent")
                stat_box.grid(row=r, column=c, sticky="w", padx=20, pady=8)
                ctk.CTkLabel(stat_box, text=f"{stat_title}:", font=("Arial", 14, "bold")).pack(side="left", padx=(0, 8))
                ctk.CTkLabel(stat_box, text=stat_val, font=("Arial", 14), text_color="#2980b9").pack(side="left")

        bottom_actions_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        bottom_actions_frame.pack(fill="x", padx=25, pady=(10, 25))

        ctk.CTkButton(
            bottom_actions_frame,
            text="📊  Export Department to Excel / CSV",
            font=("Arial", 13, "bold"),
            fg_color="#27ae60",
            hover_color="#1e8449",
            height=36,
            command=self.export_department_csv
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            bottom_actions_frame,
            text="🔄  Refresh Live Metrics",
            font=("Arial", 13),
            fg_color="#34495e",
            hover_color="#2c3e50",
            height=36,
            command=self.show_department_kpi
        ).pack(side="right")

    def export_department_csv(self):
        if not patients:
            messagebox.showwarning("Export Failed", "No patient data available to export.")
            return

        export_data = []
        for p in patients:
            score, level = calculate_risk_score(p)
            export_data.append({
                "Subject_ID": p.get("subject_id", p.get("id")),
                "Hospital_Admit_ID": p.get("hadm_id", ""),
                "ICU_Stay_ID": p.get("icustay_id", ""),
                "Room_Number": p.get("room_number", "Unassigned"),
                "Age": p.get("age"),
                "Gender": p.get("gender"),
                "Risk_Level": level,
                "Risk_Score_Percent": score,
                "Alert_Status": p.get("alert_status", "New"),
                "Heart_Rate_Mean": p.get("heart_rate_mean"),
                "Systolic_BP_Mean": p.get("systolic_bp_mean"),
                "Diastolic_BP_Mean": p.get("diastolic_bp_mean"),
                "Lactate_Max": p.get("lactate_max"),
                "Creatinine_Max": p.get("creatinine_max"),
                "Physician_Note": p.get("clinical_note", ""),
                "Last_Reviewed_At": p.get("last_reviewed_at", "Never"),
                "Reviewed_By": p.get("reviewed_by", "N/A")
            })

        df_export = pd.DataFrame(export_data)

        default_filename = f"ICU_Department_Export_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        file_path = filedialog.asksaveasfilename(
            title="Save Department Summary CSV",
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV (Comma delimited)", "*.csv"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                df_export.to_csv(file_path, index=False, encoding="utf-8-sig")
                messagebox.showinfo("Export Successful", f"Department data ({len(export_data)} patients) saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not export CSV:\n{e}")

    def show_trends(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient

        self.show_title(f"Trend Monitoring - Patient {patient['id']}")

        trends = [
            ("Heart Rate", patient.get("heart_rate_min", 0), patient.get("heart_rate_mean", 0),
             patient.get("heart_rate_max", 0), "bpm"),
            ("Systolic BP", patient.get("systolic_bp_min", 0), patient.get("systolic_bp_mean", 0),
             patient.get("systolic_bp_max", 0), "mmHg"),
            ("Diastolic BP", patient.get("diastolic_bp_min", 0), patient.get("diastolic_bp_mean", 0),
             patient.get("diastolic_bp_max", 0), "mmHg"),
        ]

        has_high_variability = False

        for name, min_v, mean_v, max_v, unit in trends:
            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=10)

            delta = float(max_v) - float(min_v)
            trend_text = f"{name}: Min={min_v} {unit} | Mean={mean_v} {unit} | Max={max_v} {unit} (Δ = {delta:.1f} {unit})"

            if delta > 40:
                has_high_variability = True
                interpretation = "⚠️ High variability detected — may indicate hemodynamic instability."
                color = "#c0392b"
            elif delta > 25:
                interpretation = "ℹ️ Moderate fluctuations observed — recommend ongoing telemetry observation."
                color = "#e67e22"
            else:
                interpretation = "✅ Stable physiological range — no significant variability detected."
                color = "#27ae60"

            ctk.CTkLabel(card, text=trend_text, font=("Arial", 17, "bold")).pack(anchor="w", padx=20, pady=(12, 4))
            ctk.CTkLabel(card, text=interpretation, text_color=color, font=("Arial", 14, "bold")).pack(anchor="w",
                                                                                                       padx=20,
                                                                                                       pady=(0, 12))

        actions_bar = ctk.CTkFrame(self.main, fg_color="transparent")
        actions_bar.pack(fill="x", padx=25, pady=(15, 25))

        def flag_instability():
            user_role = getattr(self, "logged_user", {}).get("role", "")
            if user_role != "Doctor":
                messagebox.showwarning("Permission Denied", "Only Doctors are authorized to append clinical flags.")
                return

            existing_note = patient.get("clinical_note", "")
            flag_text = f"[CLINICAL FLAG: Hemodynamic instability noted due to high vital sign variability at {datetime.datetime.now().strftime('%H:%M')}]"

            updated_note = f"{existing_note}\n{flag_text}".strip() if existing_note else flag_text
            patient["clinical_note"] = updated_note
            self.selected_patient = patient

            for p in patients:
                if str(p.get("subject_id", p.get("id"))) == str(patient.get("subject_id", patient.get("id"))):
                    p["clinical_note"] = updated_note
                    break

            save_patient_to_database(patient)
            messagebox.showinfo("Flag Recorded",
                                f"Hemodynamic alert flag has been attached to Patient {patient.get('id')} clinical note.")

        def navigate_to_what_if():
            self.set_active_button("What-if Analysis")
            self.show_what_if()

        ctk.CTkButton(
            actions_bar,
            text="🧪  Simulate Trend Stabilization in What-If",
            font=("Arial", 13, "bold"),
            fg_color="#2980b9",
            hover_color="#1f618d",
            height=36,
            command=navigate_to_what_if
        ).pack(side="right", padx=(10, 0))

        if has_high_variability:
            ctk.CTkButton(
                actions_bar,
                text="🚩  Flag Hemodynamic Instability in Chart",
                font=("Arial", 13, "bold"),
                fg_color="#c0392b",
                hover_color="#922b21",
                height=36,
                command=flag_instability
            ).pack(side="right")

    def show_missing_data(self):
        if not self.require_selected_patient():
            return
        self.clear_main()
        patient = self.selected_patient

        self.show_title(f"Missing Data & Clinical Recommendations - Patient {patient['id']}")

        missing_items = []
        if patient.get("lactate_max") is None:
            missing_items.append(
                ("Lactate", "High serum lactate is a critical marker for tissue hypoperfusion / sepsis."))
        if patient.get("creatinine_max") is None:
            missing_items.append(("Creatinine",
                                  "Serum creatinine is required to evaluate acute kidney injury (AKI) and renal clearance."))

        if not missing_items:
            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=20)

            ctk.CTkLabel(
                card,
                text="✅ Complete Clinical Profile",
                font=("Arial", 18, "bold"),
                text_color="#27ae60"
            ).pack(anchor="w", padx=20, pady=(15, 5))

            ctk.CTkLabel(
                card,
                text="All core physiological vitals and key laboratory indicators (Lactate, Creatinine) are present for this patient.",
                font=("Arial", 14)
            ).pack(anchor="w", padx=20, pady=(0, 15))

            ctk.CTkButton(
                self.main,
                text="📋  Back to Patient Summary",
                font=("Arial", 13, "bold"),
                width=220,
                height=36,
                command=self.show_patient_summary
            ).pack(pady=10)
            return

        for lab_name, clinical_impact in missing_items:
            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=10)

            ctk.CTkLabel(
                card,
                text=f"⚠️ Missing Lab Parameter: {lab_name}",
                font=("Arial", 17, "bold"),
                text_color="#e67e22"
            ).pack(anchor="w", padx=20, pady=(12, 4))

            ctk.CTkLabel(
                card,
                text=f"Clinical Relevance: {clinical_impact}",
                font=("Arial", 14),
                text_color="gray"
            ).pack(anchor="w", padx=20, pady=2)

            ctk.CTkLabel(
                card,
                text=f"Recommended Action: Order STAT {lab_name} laboratory panel to refine ML risk stratification.",
                font=("Arial", 14, "bold")
            ).pack(anchor="w", padx=20, pady=(4, 12))

        actions_bar = ctk.CTkFrame(self.main, fg_color="transparent")
        actions_bar.pack(fill="x", padx=25, pady=(15, 25))

        def order_stat_labs():
            user_role = getattr(self, "logged_user", {}).get("role", "")
            if user_role != "Doctor":
                messagebox.showwarning("Permission Denied",
                                       "Only Doctors are authorized to place STAT laboratory orders.")
                return

            missing_names = ", ".join([name for name, _ in missing_items])
            order_text = f"[STAT LAB ORDER: {missing_names} panel ordered at {datetime.datetime.now().strftime('%H:%M')} by {getattr(self, 'logged_user', {}).get('name', 'Physician')}]"

            existing_note = patient.get("clinical_note", "")
            updated_note = f"{existing_note}\n{order_text}".strip() if existing_note else order_text
            patient["clinical_note"] = updated_note
            self.selected_patient = patient

            for p in patients:
                if str(p.get("subject_id", p.get("id"))) == str(patient.get("subject_id", patient.get("id"))):
                    p["clinical_note"] = updated_note
                    break

            save_patient_to_database(patient)
            messagebox.showinfo("Lab Order Placed",
                                f"STAT Order for ({missing_names}) has been dispatched to laboratory and logged in chart.")

        ctk.CTkButton(
            actions_bar,
            text="✏️  Enter Received Lab Results",
            font=("Arial", 13, "bold"),
            fg_color="#2980b9",
            hover_color="#1f618d",
            height=36,
            command=lambda: self.edit_patient_vitals(patient)
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            actions_bar,
            text="🧪  Order STAT Missing Labs",
            font=("Arial", 13, "bold"),
            fg_color="#e67e22",
            hover_color="#d35400",
            height=36,
            command=order_stat_labs
        ).pack(side="right")

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

        ctk.CTkButton(filter_frame, text="Apply 🔍", width=90, command=apply_filters).pack(side="left", padx=15)

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

            ctk.CTkLabel(review_info_frame, text=f"⏱️ Last Review: {last_review} (by {reviewer})", font=("Arial", 13), text_color="gray").pack(side="left", padx=5)
            ctk.CTkLabel(review_info_frame, text=reassess_text, font=("Arial", 13, "bold"), text_color=reassess_color).pack(side="right", padx=5)

            button_frame = ctk.CTkFrame(card, fg_color="transparent")
            button_frame.pack(anchor="e", padx=15, pady=10)

            ctk.CTkButton(button_frame, text="Open 👤", width=80, command=lambda p=patient: self.select_patient(p)).pack(side="left", padx=4)
            ctk.CTkButton(button_frame, text="Reviewed 👁️", width=95, command=lambda p=patient: self.update_alert_status(p, "Reviewed")).pack(side="left", padx=4)
            ctk.CTkButton(button_frame, text="In Progress ⏳", width=105, command=lambda p=patient: self.update_alert_status(p, "In Progress")).pack(side="left", padx=4)
            ctk.CTkButton(button_frame, text="Resolved ✅", width=90, command=lambda p=patient: self.update_alert_status(p, "Resolved")).pack(side="left", padx=4)

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
        messagebox.showinfo("Alert Updated", f"Patient {patient.get('id', 'N/A')} alert marked as {status}.")
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

        options_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        options_frame.pack(pady=10)

        ctk.CTkLabel(options_frame, text="Gender", width=150, anchor="w").grid(row=0, column=0, padx=10, pady=8)
        gender_var = ctk.StringVar(value="Female")
        gender_menu = ctk.CTkOptionMenu(options_frame, values=["Female", "Male"], variable=gender_var, width=180)
        gender_menu.grid(row=0, column=1, padx=10, pady=8)

        ctk.CTkLabel(options_frame, text="Admission Time", width=150, anchor="w").grid(row=1, column=0, padx=10, pady=8)
        time_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        time_frame.grid(row=1, column=1, padx=10, pady=8)

        admit_hour_var = ctk.StringVar(value="12")
        admit_minute_var = ctk.StringVar(value="00")

        hour_menu = ctk.CTkOptionMenu(time_frame, values=[f"{i:02d}" for i in range(24)], variable=admit_hour_var, width=80)
        hour_menu.pack(side="left")

        ctk.CTkLabel(time_frame, text=":", font=("Arial", 18, "bold")).pack(side="left", padx=5)

        minute_menu = ctk.CTkOptionMenu(time_frame, values=["00", "15", "30", "45"], variable=admit_minute_var, width=80)
        minute_menu.pack(side="left")

        ctk.CTkLabel(options_frame, text="Admission Day", width=150, anchor="w").grid(row=2, column=0, padx=10, pady=8)
        admit_day_var = ctk.StringVar(value="Monday")

        day_menu = ctk.CTkOptionMenu(
            options_frame,
            values=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            variable=admit_day_var, width=180
        )
        day_menu.grid(row=2, column=1, padx=10, pady=8)

        def save_patient():
            try:
                subject_id = entries["subject_id"].get().strip()
                hadm_id = entries["hadm_id"].get().strip()
                icustay_id = entries["icustay_id"].get().strip()

                if not subject_id or not hadm_id or not icustay_id:
                    raise ValueError("Subject ID, Admission ID, and ICU Stay ID are required.")

                for existing_patient in patients:
                    if str(existing_patient.get("icustay_id", "")) == icustay_id:
                        raise ValueError(f"ICU Stay ID {icustay_id} already exists.")

                required_numeric = [
                    "age", "heart_rate_mean", "heart_rate_min", "heart_rate_max",
                    "systolic_bp_mean", "systolic_bp_min", "systolic_bp_max",
                    "diastolic_bp_mean", "diastolic_bp_min", "diastolic_bp_max"
                ]

                values = {}
                for field in required_numeric:
                    raw_value = entries[field].get().strip()
                    if not raw_value:
                        raise ValueError(f"{field.replace('_', ' ').title()} is required.")
                    try:
                        values[field] = float(raw_value)
                    except ValueError:
                        raise ValueError(f"{field.replace('_', ' ').title()} must be numeric.")

                if values["age"] < 0 or values["age"] > 120:
                    raise ValueError("Age must be between 0 and 120.")

                vital_fields = [
                    "heart_rate_mean", "heart_rate_min", "heart_rate_max",
                    "systolic_bp_mean", "systolic_bp_min", "systolic_bp_max",
                    "diastolic_bp_mean", "diastolic_bp_min", "diastolic_bp_max"
                ]

                for field in vital_fields:
                    if values[field] <= 0:
                        raise ValueError(f"{field.replace('_', ' ').title()} must be greater than 0.")

                if not (values["heart_rate_min"] <= values["heart_rate_mean"] <= values["heart_rate_max"]):
                    raise ValueError("Heart Rate must satisfy: Min ≤ Mean ≤ Max.")

                if not (values["systolic_bp_min"] <= values["systolic_bp_mean"] <= values["systolic_bp_max"]):
                    raise ValueError("Systolic BP must satisfy: Min ≤ Mean ≤ Max.")

                if not (values["diastolic_bp_min"] <= values["diastolic_bp_mean"] <= values["diastolic_bp_max"]):
                    raise ValueError("Diastolic BP must satisfy: Min ≤ Mean ≤ Max.")

                creatinine = self.optional_float(entries["creatinine_max"].get())
                lactate = self.optional_float(entries["lactate_max"].get())

                if creatinine is not None and creatinine < 0:
                    raise ValueError("Creatinine cannot be negative.")
                if lactate is not None and lactate < 0:
                    raise ValueError("Lactate cannot be negative.")

                gender = "F" if gender_var.get() == "Female" else "M"
                admit_hour = int(admit_hour_var.get())
                day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
                admit_day = day_mapping[admit_day_var.get()]

                patient = {
                    "id": subject_id, "subject_id": subject_id, "hadm_id": hadm_id, "icustay_id": icustay_id,
                    "age": values["age"], "gender": gender,
                    "heart_rate_mean": values["heart_rate_mean"], "heart_rate_min": values["heart_rate_min"], "heart_rate_max": values["heart_rate_max"],
                    "systolic_bp_mean": values["systolic_bp_mean"], "systolic_bp_min": values["systolic_bp_min"], "systolic_bp_max": values["systolic_bp_max"],
                    "diastolic_bp_mean": values["diastolic_bp_mean"], "diastolic_bp_min": values["diastolic_bp_min"], "diastolic_bp_max": values["diastolic_bp_max"],
                    "creatinine_max": creatinine, "lactate_max": lactate,
                    "admit_hour": admit_hour, "admit_dayofweek": admit_day,
                    "icu_hours": 0, "admission_type": "Manual Entry", "diagnosis": "Not provided",
                    "alert_status": "New", "clinical_note": "", "room_number": f"ICU-{(len(patients) % 15) + 101}"
                }

                result = call_model(patient)
                patients.append(patient)
                save_patient_to_database(patient)
                self.selected_patient = patient

                messagebox.showinfo(
                    "Patient Added",
                    f"Patient {patient['id']} was added successfully.\n\nRisk level: {result['risk_level']}\nRisk: {result['risk_percent']}"
                )
                self.show_patient_summary()

            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Could not add patient:\n{e}")

        ctk.CTkButton(self.main, text="➕ Add Patient & Calculate Risk 📊", width=280, height=45, command=save_patient).pack(pady=25)

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

            minutes, seconds = int(remaining // 60), int(remaining % 60)
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
                all_new_patients, skipped, BATCH_SIZE = [], 0, 500

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
                                "id": str(row["SUBJECT_ID"]), "subject_id": str(row["SUBJECT_ID"]), "hadm_id": str(row["HADM_ID"]), "icustay_id": str(row["ICUSTAY_ID"]),
                                "intime": intime_text, "age": float(row["AGE"]), "gender": str(row["GENDER"]).strip().upper(),
                                "heart_rate_mean": float(row["HeartRate_mean"]), "heart_rate_min": float(row["HeartRate_min"]), "heart_rate_max": float(row["HeartRate_max"]),
                                "systolic_bp_mean": float(row["SysBP_mean"]), "systolic_bp_min": float(row["SysBP_min"]), "systolic_bp_max": float(row["SysBP_max"]),
                                "diastolic_bp_mean": float(row["DiasBP_mean"]), "diastolic_bp_min": float(row["DiasBP_min"]), "diastolic_bp_max": float(row["DiasBP_max"]),
                                "creatinine_max": None if pd.isna(row.get("Creatinine_max")) else float(row["Creatinine_max"]),
                                "lactate_max": None if pd.isna(row.get("Lactate_max")) else float(row["Lactate_max"]),
                                "admit_hour": admit_hour, "admit_dayofweek": admit_day, "icu_hours": "N/A",
                                "admission_type": "CSV Import", "diagnosis": "Not provided", "alert_status": "New", "clinical_note": "",
                                "room_number": f"ICU-{(idx % 15) + 101}"
                            }

                            if "HOSPITAL_EXPIRE_FLAG" in df.columns:
                                val = row.get("HOSPITAL_EXPIRE_FLAG")
                                patient["hospital_expire_flag"] = None if pd.isna(val) else int(val)

                            batch_patients.append(patient)
                        except Exception:
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


if __name__ == "__main__":
    initialize_application_data()

    while True:
        login = LoginWindow()
        login.mainloop()

        logged_user = getattr(login, "logged_user", None)

        try:
            login.destroy()
        except Exception:
            pass

        if logged_user is None:
            break

        app = ICUApp(logged_user)
        app.mainloop()

        switch_requested = getattr(app, "switch_user_requested", False)

        try:
            app.destroy()
        except Exception:
            pass

        if switch_requested:
            continue

        break