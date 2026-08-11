import pytest
import app


@pytest.fixture
def test_database(tmp_path, monkeypatch):

    # Create a completely temporary SQLite file
    temp_db = tmp_path / "test_icu_system.db"

    # Tell app.py to use this DB instead of icu_system.db
    monkeypatch.setattr(
        app,
        "DATABASE_PATH",
        temp_db
    )

    # Create tables inside temporary database
    app.init_database()

    # Reset global patients list
    app.patients.clear()

    yield temp_db

    # tmp_path is automatically cleaned by pytest