import sqlite3
import json
import os
from datetime import datetime, timezone


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "sentinelai.db"
)


def get_connection():

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS analysis_logs (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                text TEXT NOT NULL,
                source TEXT NOT NULL,

                prediction TEXT NOT NULL,

                benign_probability REAL,
                malicious_probability REAL,

                risk_score INTEGER,
                risk_level TEXT,

                detected_threats TEXT,

                is_untrusted_source BOOLEAN,

                recommended_action TEXT,

                created_at TEXT NOT NULL
            )
        """)


def save_analysis(text, source, result):

    with get_connection() as connection:

        connection.execute("""
            INSERT INTO analysis_logs (
                text,
                source,
                prediction,
                benign_probability,
                malicious_probability,
                risk_score,
                risk_level,
                detected_threats,
                is_untrusted_source,
                recommended_action,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            text[:500],
            source,
            result["prediction"],
            result["benign_probability"],
            result["malicious_probability"],
            result["risk_score"],
            result["risk_level"],
            json.dumps(result["detected_threats"]),
            result["is_untrusted_source"],
            result["recommended_action"],
            datetime.now(timezone.utc).isoformat()
        ))


def format_analysis_log(row):

    if row is None:
        return None

    log = dict(row)

    try:
        log["detected_threats"] = json.loads(
            log["detected_threats"] or "[]"
        )
    except (json.JSONDecodeError, TypeError):
        log["detected_threats"] = []

    log["is_untrusted_source"] = bool(
        log["is_untrusted_source"]
    )

    return log


def get_analysis_logs(limit=50):

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT * FROM analysis_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [
        format_analysis_log(row)
        for row in rows
    ]


def get_analysis_by_id(analysis_id):

    with get_connection() as connection:

        row = connection.execute("""
            SELECT *
            FROM analysis_logs
            WHERE id = ?
        """, (analysis_id,)).fetchone()

    return format_analysis_log(row)


def delete_analysis(analysis_id):

    with get_connection() as connection:

        cursor = connection.execute("""
            DELETE FROM analysis_logs
            WHERE id = ?
        """, (analysis_id,))

        return cursor.rowcount > 0


def get_statistics():

    with get_connection() as connection:

        total = connection.execute("""
            SELECT COUNT(*) FROM analysis_logs
        """).fetchone()[0]

        malicious = connection.execute("""
            SELECT COUNT(*) FROM analysis_logs
            WHERE prediction = 'MALICIOUS'
        """).fetchone()[0]

        blocked = connection.execute("""
            SELECT COUNT(*) FROM analysis_logs
            WHERE recommended_action = 'BLOCK'
        """).fetchone()[0]

        flagged = connection.execute("""
            SELECT COUNT(*) FROM analysis_logs
            WHERE recommended_action = 'FLAG_FOR_REVIEW'
        """).fetchone()[0]

    return {
        "total_scans": total,
        "malicious_detections": malicious,
        "blocked_requests": blocked,
        "flagged_requests": flagged,
        "safe_scans": total - malicious
    }

def get_all_analysis_logs():

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT *
            FROM analysis_logs
            ORDER BY id DESC
        """).fetchall()

    return [
        format_analysis_log(row)
        for row in rows
    ]