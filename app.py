import customtkinter as ctk
from tkinter import messagebox
import copy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from app import compute_features, generate_alerts, get_risk_drivers, model, scaler, FEATURES, THRESHOLD
import pandas as pd
import numpy as np

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


patients = [
    {
        "id": "P1001", "age": 72, "gender": "F", "icu_hours": 8,
        "heart_rate_mean": 118, "heart_rate_min": 95, "heart_rate_max": 138,
        "systolic_bp_mean": 92, "systolic_bp_min": 82, "systolic_bp_max": 110,
        "diastolic_bp_mean": 58, "diastolic_bp_min": 48, "diastolic_bp_max": 70,
        "lactate_max": 3.2, "creatinine_max": 1.8,
        "admit_hour": 14, "admit_dayofweek": 2,
        "admission_type": "Emergency",
        "diagnosis": "Sepsis suspicion",
        "alert_status": "New",
        "clinical_note": ""
    },
    {
        "id": "P1002", "age": 55, "gender": "M", "icu_hours": 5,
        "heart_rate_mean": 88, "heart_rate_min": 72, "heart_rate_max": 101,
        "systolic_bp_mean": 118, "systolic_bp_min": 105, "systolic_bp_max": 132,
        "diastolic_bp_mean": 74, "diastolic_bp_min": 65, "diastolic_bp_max": 82,
        "lactate_max": 1.4, "creatinine_max": 1.0,
        "admit_hour": 10, "admit_dayofweek": 3,
        "admission_type": "Urgent",
        "diagnosis": "Respiratory infection",
        "alert_status": "New",
        "clinical_note": ""
    },
    {
        "id": "P1003", "age": 81, "gender": "M", "icu_hours": 11,
        "heart_rate_mean": 126, "heart_rate_min": 102, "heart_rate_max": 150,
        "systolic_bp_mean": 88, "systolic_bp_min": 75, "systolic_bp_max": 104,
        "diastolic_bp_mean": 50, "diastolic_bp_min": 42, "diastolic_bp_max": 62,
        "lactate_max": None, "creatinine_max": 2.4,
        "admit_hour": 2, "admit_dayofweek": 6,
        "admission_type": "Emergency",
        "diagnosis": "Cardiac complication",
        "alert_status": "New",
        "clinical_note": ""
    }
]


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


def calculate_risk_score(patient):
    try:
        result = call_model(patient)
        score = int(result["risk_score"] * 100)
        level = result["risk_level"]
        return score, level
    except Exception as e:
        print("Model connection error:", e)
        return 0, "ERROR"


def get_risk_color(level):
    colors = {
        "CRITICAL": "#c0392b",
        "HIGH": "#e67e22",
        "MODERATE": "#f1c40f",
        "LOW": "#27ae60",
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


class ICUApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ICU Clinical Decision Support System")
        self.geometry("1350x800")
        self.selected_patient = patients[0]

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
            text="ICU Decision\nSupport System",
            font=("Arial", 23, "bold")
        ).pack(pady=25)

        buttons = [
            ("Dashboard", self.show_dashboard),
            ("Priority Queue", self.show_priority_queue),
            ("Patient Summary", self.show_patient_summary),
            ("Patient Details", self.show_patient_details),
            ("Risk Explanation", self.show_risk_explanation),
            ("Recommended Actions", self.show_recommended_actions),
            ("Alerts Workflow", self.show_alerts),
            ("What-if Analysis", self.show_what_if),
            ("Trend Monitoring", self.show_trends),
            ("Missing Data", self.show_missing_data),
        ]

        for text, command in buttons:
            ctk.CTkButton(self.sidebar, text=text, command=command).pack(
                pady=6, fill="x", padx=18
            )

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
        self.show_patient_summary()

    # 1. Dashboard
    def show_dashboard(self):
        self.clear_main()
        self.show_title("ICU Dashboard")

        frame = ctk.CTkFrame(self.main)
        frame.pack(fill="x", padx=20, pady=10)

        headers = ["Patient", "Age", "Gender", "ICU Hours", "Risk Score", "Risk Level", "Action"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(frame, text=header, font=("Arial", 14, "bold")).grid(row=0, column=col, padx=15, pady=10)

        for row, patient in enumerate(patients, start=1):
            score, level = calculate_risk_score(patient)

            values = [
                patient.get("id"),
                patient.get("age"),
                patient.get("gender"),
                patient.get("icu_hours", "N/A"),
                f"{score}/100",
                level
            ]

            for col, value in enumerate(values):
                if col == 5:
                    ctk.CTkLabel(
                        frame,
                        text=value,
                        fg_color=get_risk_color(level),
                        text_color="white",
                        corner_radius=8,
                        width=100
                    ).grid(row=row, column=col, padx=15, pady=8)
                else:
                    ctk.CTkLabel(frame, text=str(value), font=("Arial", 13)).grid(row=row, column=col, padx=15, pady=8)

            ctk.CTkButton(
                frame,
                text="Open",
                width=80,
                command=lambda p=patient: self.select_patient(p)
            ).grid(row=row, column=6, padx=15, pady=8)

    # 2. Priority Queue
    def show_priority_queue(self):
        self.clear_main()
        self.show_title("Clinical Priority Queue")

        ranked = []
        for patient in patients:
            score, level = calculate_risk_score(patient)
            ranked.append((score, level, patient))

        ranked.sort(reverse=True, key=lambda x: x[0])

        for i, (score, level, patient) in enumerate(ranked, start=1):
            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=10)

            ctk.CTkLabel(
                card,
                text=f"{i}. Patient {patient['id']} — {level} risk — {score}/100",
                font=("Arial", 20, "bold"),
                text_color=get_risk_color(level)
            ).pack(anchor="w", padx=20, pady=10)

            action = get_recommended_actions(level)[0]
            ctk.CTkLabel(
                card,
                text=f"Suggested priority: {action}",
                font=("Arial", 15)
            ).pack(anchor="w", padx=20, pady=5)

            ctk.CTkButton(
                card,
                text="Open Patient",
                command=lambda p=patient: self.select_patient(p)
            ).pack(anchor="e", padx=20, pady=10)

    # 8. Patient Summary Card
    def show_patient_summary(self):
        self.clear_main()
        patient = self.selected_patient
        score, level = calculate_risk_score(patient)
        result = patient.get("model_result", {})

        self.show_title(f"Patient Summary Card - {patient['id']}")

        card = ctk.CTkFrame(self.main)
        card.pack(fill="x", padx=25, pady=15)

        summary = [
            f"Risk level: {level}",
            f"Risk score: {score}/100",
            f"Prediction: {result.get('prediction', 'N/A')}",
            f"Admission type: {patient.get('admission_type', 'N/A')}",
            f"Diagnosis: {patient.get('diagnosis', 'N/A')}",
            f"Alert status: {patient.get('alert_status', 'New')}",
        ]

        for line in summary:
            ctk.CTkLabel(card, text=line, font=("Arial", 17)).pack(anchor="w", padx=25, pady=6)

        ctk.CTkLabel(card, text="Suggested next steps:", font=("Arial", 18, "bold")).pack(anchor="w", padx=25, pady=12)

        for action in get_recommended_actions(level):
            ctk.CTkLabel(card, text=f"• {action}", font=("Arial", 15)).pack(anchor="w", padx=45, pady=3)

    # Patient Details
    def show_patient_details(self):
        self.clear_main()
        patient = self.selected_patient
        self.show_title(f"Patient Details - {patient['id']}")

        details = [
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

    # 4. Explanation Panel
    def show_risk_explanation(self):
        self.clear_main()
        patient = self.selected_patient
        calculate_risk_score(patient)
        result = patient.get("model_result", {})

        self.show_title(f"Why This Risk Score? - {patient['id']}")

        drivers = result.get("risk_drivers", [])
        if not drivers:
            ctk.CTkLabel(self.main, text="No risk drivers returned by the model.", font=("Arial", 17)).pack(pady=20)
            return

        for driver in drivers:
            direction = driver.get("direction")
            arrow = "▲ raises risk" if direction == "increase" else "▼ lowers risk"
            color = "#c0392b" if direction == "increase" else "#27ae60"

            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=8)

            ctk.CTkLabel(
                card,
                text=f"{driver.get('name')} — {driver.get('value')}",
                font=("Arial", 18, "bold")
            ).pack(anchor="w", padx=20, pady=6)

            ctk.CTkLabel(
                card,
                text=f"{arrow} | impact: {driver.get('impact')}",
                text_color=color,
                font=("Arial", 15)
            ).pack(anchor="w", padx=20, pady=4)

    # 2. Recommended Actions
    def show_recommended_actions(self):
        self.clear_main()
        patient = self.selected_patient
        score, level = calculate_risk_score(patient)

        self.show_title(f"Recommended Actions - {patient['id']}")

        ctk.CTkLabel(
            self.main,
            text=f"Current risk level: {level} ({score}/100)",
            font=("Arial", 22, "bold"),
            text_color=get_risk_color(level)
        ).pack(pady=10)

        for action in get_recommended_actions(level):
            card = ctk.CTkFrame(self.main)
            card.pack(fill="x", padx=25, pady=8)
            ctk.CTkLabel(card, text=f"• {action}", font=("Arial", 17)).pack(anchor="w", padx=25, pady=12)

        ctk.CTkLabel(
            self.main,
            text="Note: recommendations are for decision support only and do not replace clinical judgment.",
            text_color="gray",
            font=("Arial", 14)
        ).pack(pady=20)

    # 3. Alert Workflow
    def show_alerts(self):
        self.clear_main()
        self.show_title("Alert Workflow")

        for patient in patients:
            score, level = calculate_risk_score(patient)
            result = patient.get("model_result", {})
            alerts = result.get("clinical_alerts", [])

            if level in ["HIGH", "CRITICAL", "MODERATE"] or alerts:
                card = ctk.CTkFrame(self.main)
                card.pack(fill="x", padx=25, pady=10)

                ctk.CTkLabel(
                    card,
                    text=f"Patient {patient['id']} | {level} | Status: {patient.get('alert_status', 'New')}",
                    font=("Arial", 19, "bold")
                ).pack(anchor="w", padx=20, pady=8)

                for alert in alerts[:3]:
                    ctk.CTkLabel(
                        card,
                        text=f"{alert.get('icon', '')} {alert.get('text')}",
                        font=("Arial", 14)
                    ).pack(anchor="w", padx=25, pady=3)

                buttons = ctk.CTkFrame(card)
                buttons.pack(anchor="e", padx=20, pady=10)

                ctk.CTkButton(buttons, text="Reviewed", command=lambda p=patient: self.update_alert_status(p, "Reviewed")).pack(side="left", padx=5)
                ctk.CTkButton(buttons, text="In Progress", command=lambda p=patient: self.update_alert_status(p, "In Progress")).pack(side="left", padx=5)
                ctk.CTkButton(buttons, text="Resolved", command=lambda p=patient: self.update_alert_status(p, "Resolved")).pack(side="left", padx=5)

    def update_alert_status(self, patient, status):
        patient["alert_status"] = status
        messagebox.showinfo("Alert Updated", f"Patient {patient['id']} alert marked as {status}.")
        self.show_alerts()

    # 5. What-if Scenario
    def show_what_if(self):
        self.clear_main()
        patient = self.selected_patient
        original_score, original_level = calculate_risk_score(patient)

        self.show_title(f"What-if Analysis - {patient['id']}")

        ctk.CTkLabel(
            self.main,
            text=f"Original risk: {original_level} ({original_score}/100)",
            font=("Arial", 20, "bold")
        ).pack(pady=10)

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

                result_label.configure(
                    text=f"New risk: {new_level} ({new_score}/100)"
                )
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ctk.CTkButton(self.main, text="Run What-if Analysis", command=run_what_if).pack(pady=10)

    # 6. Trend Monitoring
    def show_trends(self):
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

    # 7. Missing Data Recommendations
    def show_missing_data(self):
        self.clear_main()
        patient = self.selected_patient

        self.show_title(f"Missing Data Recommendations - {patient['id']}")

        missing = []

        if patient.get("lactate_max") is None:
            missing.append("Lactate is missing — consider ordering lactate test if clinically relevant.")
        if patient.get("creatinine_max") is None:
            missing.append("Creatinine is missing — consider reviewing renal function labs.")

        if not missing:
            ctk.CTkLabel(
                self.main,
                text="No important missing data detected.",
                font=("Arial", 18),
                text_color="#27ae60"
            ).pack(pady=30)
        else:
            for item in missing:
                card = ctk.CTkFrame(self.main)
                card.pack(fill="x", padx=25, pady=10)
                ctk.CTkLabel(card, text=f"• {item}", font=("Arial", 16)).pack(anchor="w", padx=20, pady=12)


if __name__ == "__main__":
    app = ICUApp()
    app.mainloop()