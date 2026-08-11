from app import (
    hash_password,
    get_missing_model_fields,
    authenticate_user
)

import app
import sqlite3
import pytest


def create_test_patient():
    return {
        "id": "999",
        "subject_id": "999",
        "hadm_id": "1001",
        "icustay_id": "2001",

        "intime": "",

        "gender": "M",
        "age": 60,

        "heart_rate_mean": 85,
        "heart_rate_min": 70,
        "heart_rate_max": 100,

        "systolic_bp_mean": 120,
        "systolic_bp_min": 105,
        "systolic_bp_max": 135,

        "diastolic_bp_mean": 75,
        "diastolic_bp_min": 65,
        "diastolic_bp_max": 85,

        "creatinine_max": 1.0,
        "lactate_max": 1.5,

        "admit_hour": 10,
        "admit_dayofweek": 2,

        "icu_hours": 0,

        "admission_type": "Test",
        "diagnosis": "Test",

        "alert_status": "New",
        "clinical_note": "",

        "hospital_expire_flag": None,

        "room_number": "ICU-101"
    }

def test_patient_is_saved_and_loaded(test_database):

    patient = create_test_patient()

    app.save_patient_to_database(patient)

    loaded_patients = (
        app.load_patients_from_database()
    )

    assert len(loaded_patients) == 1

    loaded = loaded_patients[0]

    assert loaded["subject_id"] == "999"
    assert loaded["hadm_id"] == "1001"
    assert loaded["icustay_id"] == "2001"

    assert loaded["age"] == 60
    assert loaded["gender"] == "M"

    assert loaded["room_number"] == "ICU-101"

    # This patient was saved before any risk calculation,
    # so no cached prediction should exist.
    assert "_risk_score" not in loaded
    assert "_risk_level" not in loaded

def test_patient_persists_after_reload(test_database):

    patient = create_test_patient()

    app.save_patient_to_database(patient)

    # Simulate closing / reopening the application
    app.patients.clear()

    reloaded = app.load_patients_from_database()

    assert len(reloaded) == 1
    assert reloaded[0]["subject_id"] == "999"

def test_same_icu_stay_is_not_duplicated(
    test_database
):
    patient = create_test_patient()

    app.save_patient_to_database(patient)

    patient["age"] = 61

    app.save_patient_to_database(patient)

    loaded = app.load_patients_from_database()

    assert len(loaded) == 1
    assert loaded[0]["age"] == 61

def test_password_hash_is_consistent():
    password = "doctor123"

    result1 = hash_password(password)
    result2 = hash_password(password)

    assert result1 == result2

def test_different_passwords_have_different_hashes():
    assert hash_password("doctor123") != hash_password("wrongpassword")

def test_valid_doctor_login():
    user = authenticate_user(
        "doctor1",
        "doctor123"
    )

    assert user is not None
    assert user["username"] == "doctor1"
    assert user["role"] == "Doctor"

def test_wrong_password_rejected():
    user = authenticate_user(
        "doctor1",
        "WRONG_PASSWORD"
    )

    assert user is None

# ==========================================================
# PASSWORD TESTS
# ==========================================================

def test_same_password_produces_same_hash():
    first = app.hash_password("doctor123")
    second = app.hash_password("doctor123")

    assert first == second


def test_different_passwords_produce_different_hashes():
    first = app.hash_password("doctor123")
    second = app.hash_password("wrongPassword")

    assert first != second


def test_password_is_not_stored_as_plain_text():
    password = "doctor123"

    hashed = app.hash_password(password)

    assert hashed != password


# ==========================================================
# MISSING DATA TESTS
# ==========================================================

def test_complete_patient_has_no_missing_model_fields():
    patient = create_test_patient()

    missing = app.get_missing_model_fields(patient)

    assert missing == []


def test_missing_heart_rate_is_detected():
    patient = create_test_patient()

    patient["heart_rate_mean"] = None

    missing = app.get_missing_model_fields(patient)

    assert "heart_rate_mean" in missing


def test_nan_value_is_detected_as_missing():
    patient = create_test_patient()

    patient["systolic_bp_min"] = float("nan")

    missing = app.get_missing_model_fields(patient)

    assert "systolic_bp_min" in missing


def test_optional_lactate_can_be_missing():
    patient = create_test_patient()

    patient["lactate_max"] = None

    missing = app.get_missing_model_fields(patient)

    assert "lactate_max" not in missing


def test_optional_creatinine_can_be_missing():
    patient = create_test_patient()

    patient["creatinine_max"] = None

    missing = app.get_missing_model_fields(patient)

    assert "creatinine_max" not in missing


# ==========================================================
# RISK CALCULATION TESTS
# ==========================================================

def test_incomplete_patient_gets_incomplete_status():
    patient = create_test_patient()

    patient["heart_rate_mean"] = None

    score, level = app.calculate_risk_score(patient)

    assert score == 0
    assert level == "INCOMPLETE"


def test_incomplete_patient_saves_missing_fields():
    patient = create_test_patient()

    patient["heart_rate_mean"] = None

    app.calculate_risk_score(patient)

    assert "_missing_model_fields" in patient
    assert "heart_rate_mean" in patient["_missing_model_fields"]


def test_cached_risk_score_is_reused(monkeypatch):
    patient = create_test_patient()

    patient["_risk_score"] = 0.72
    patient["_risk_level"] = "CRITICAL"

    def forbidden_model_call(patient):
        raise AssertionError(
            "Model should not be called when prediction is cached."
        )

    monkeypatch.setattr(
        app,
        "call_model",
        forbidden_model_call
    )

    score, level = app.calculate_risk_score(patient)

    assert score == 72
    assert level == "CRITICAL"


def test_model_result_is_cached(monkeypatch):
    patient = create_test_patient()

    fake_result = {
        "risk_score": 0.45,
        "risk_level": "HIGH"
    }

    def fake_call_model(patient):
        return fake_result

    monkeypatch.setattr(
        app,
        "call_model",
        fake_call_model
    )

    score, level = app.calculate_risk_score(patient)

    assert score == 45
    assert level == "HIGH"

    assert patient["_risk_score"] == 0.45
    assert patient["_risk_level"] == "HIGH"


def test_model_failure_returns_error(monkeypatch):
    patient = create_test_patient()

    def broken_model(patient):
        raise RuntimeError("Test model failure")

    monkeypatch.setattr(
        app,
        "call_model",
        broken_model
    )

    score, level = app.calculate_risk_score(patient)

    assert score == 0
    assert level == "ERROR"

# ==========================================================
# DATABASE TESTS
# ==========================================================

def test_multiple_patients_can_be_saved(test_database):

    patient1 = create_test_patient()

    patient2 = create_test_patient()
    patient2["id"] = "888"
    patient2["subject_id"] = "888"
    patient2["hadm_id"] = "1002"
    patient2["icustay_id"] = "2002"

    app.save_patient_to_database(patient1)
    app.save_patient_to_database(patient2)

    loaded = app.load_patients_from_database()

    assert len(loaded) == 2


def test_patient_update_is_persisted(test_database):

    patient = create_test_patient()

    app.save_patient_to_database(patient)

    patient["room_number"] = "ICU-115"

    app.save_patient_to_database(patient)

    loaded = app.load_patients_from_database()

    assert len(loaded) == 1
    assert loaded[0]["room_number"] == "ICU-115"


def test_clinical_note_is_persisted(test_database):

    patient = create_test_patient()

    patient["clinical_note"] = (
        "Patient requires further clinical review."
    )

    app.save_patient_to_database(patient)

    loaded = app.load_patients_from_database()

    assert (
        loaded[0]["clinical_note"]
        == "Patient requires further clinical review."
    )


def test_alert_status_is_persisted(test_database):

    patient = create_test_patient()

    patient["alert_status"] = "Reviewed"

    app.save_patient_to_database(patient)

    loaded = app.load_patients_from_database()

    assert loaded[0]["alert_status"] == "Reviewed"


def test_risk_prediction_is_persisted(test_database):

    patient = create_test_patient()

    patient["_risk_score"] = 0.63
    patient["_risk_level"] = "CRITICAL"

    app.save_patient_to_database(patient)

    loaded = app.load_patients_from_database()

    assert loaded[0]["_risk_score"] == pytest.approx(0.63)
    assert loaded[0]["_risk_level"] == "CRITICAL"

# ==========================================================
# USER / AUTHENTICATION TESTS
# ==========================================================

def test_new_user_can_login(test_database):

    app.add_user_to_database(
        "testdoctor",
        "test123",
        "Dr. Test User",
        "Doctor"
    )

    user = app.authenticate_user(
        "testdoctor",
        "test123"
    )

    assert user is not None

    assert user["username"] == "testdoctor"
    assert user["name"] == "Dr. Test User"
    assert user["role"] == "Doctor"


def test_wrong_password_is_rejected(test_database):

    app.add_user_to_database(
        "testdoctor",
        "test123",
        "Dr. Test User",
        "Doctor"
    )

    user = app.authenticate_user(
        "testdoctor",
        "wrongPassword"
    )

    assert user is None


def test_unknown_user_is_rejected(test_database):

    user = app.authenticate_user(
        "does_not_exist",
        "password"
    )

    assert user is None


def test_deleted_user_cannot_login(test_database):

    app.add_user_to_database(
        "temporary",
        "password123",
        "Temporary User",
        "Nurse"
    )

    # Confirm that account initially works
    assert (
        app.authenticate_user(
            "temporary",
            "password123"
        )
        is not None
    )

    app.delete_user_from_database(
        "temporary"
    )

    # Account should now be gone
    assert (
        app.authenticate_user(
            "temporary",
            "password123"
        )
        is None
    )


def test_admin_role_is_preserved(test_database):

    app.add_user_to_database(
        "testadmin",
        "admin123",
        "Test Administrator",
        "Admin"
    )

    user = app.authenticate_user(
        "testadmin",
        "admin123"
    )

    assert user["role"] == "Admin"


def test_duplicate_username_is_rejected(test_database):

    app.add_user_to_database(
        "duplicate",
        "password1",
        "First User",
        "Doctor"
    )

    with pytest.raises(sqlite3.IntegrityError):

        app.add_user_to_database(
            "duplicate",
            "password2",
            "Second User",
            "Nurse"
        )

# ==========================================

def test_model_prediction_is_deterministic():
    patient1 = create_test_patient()
    patient2 = create_test_patient()

    result1 = app.call_model(patient1)
    result2 = app.call_model(patient2)

    assert (
        result1["risk_score"]
        == result2["risk_score"]
    )

    assert (
        result1["risk_level"]
        == result2["risk_level"]
    )