"""Tiny SQLite store for patient assessments (local file data/patients.db).
NOTE: on free cloud hosts (e.g. Streamlit Community Cloud) the disk is temporary, so records
can disappear when the app restarts. Locally the file is permanent."""
import os, json, sqlite3
from datetime import datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "patients.db")


def _conn():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT, patient_code TEXT, name TEXT, created_at TEXT,
        age INTEGER, sex TEXT, probability REAL, flagged INTEGER, risk_band TEXT,
        features TEXT, top_factors TEXT, notes TEXT, source TEXT)""")
    return c


def add_patient(name, features, result, band, notes="", source="user", created_at=None):
    with _conn() as c:
        cur = c.execute("INSERT INTO patients(name,created_at,age,sex,probability,flagged,risk_band,features,top_factors,notes,source) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (name or "Anonymous", (created_at or datetime.now()).isoformat(timespec="seconds"), int(features["age"]),
                         "Male" if features.get("male") == 1 else "Female", result["probability_10y_chd"],
                         int(result["flagged_elevated_risk"]), band, json.dumps(features),
                         json.dumps(result["top_contributions"]), notes, source))
        pid = cur.lastrowid
        c.execute("UPDATE patients SET patient_code=? WHERE id=?", (f"P{pid:04d}", pid))
    return pid


def all_patients() -> pd.DataFrame:
    with _conn() as c:
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY id DESC", c)
    if len(df):
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def delete_patient(pid):
    with _conn() as c:
        c.execute("DELETE FROM patients WHERE id=?", (int(pid),))


def clear_all(only_demo=False):
    with _conn() as c:
        c.execute("DELETE FROM patients WHERE source='demo'" if only_demo else "DELETE FROM patients")
