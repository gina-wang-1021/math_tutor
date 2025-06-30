import sqlite3
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..')) # This will point to project root when script is in scripts/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .logger_config import setup_logger

logger = setup_logger('build_historic_store')

DB_SCHEMA = """
DROP TABLE IF EXISTS qa_pairs;
CREATE TABLE IF NOT EXISTS qa_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_text TEXT NOT NULL UNIQUE,
    answer_text TEXT NOT NULL
);
"""

# Define DB_DIR relative to the project root
DB_DIR = os.path.join(_PROJECT_ROOT, "historic_qa_data")
GRADE_DBS = {
    11: os.path.join(DB_DIR, "grade11_historic.db"),
    12: os.path.join(DB_DIR, "grade12_historic.db")
}

def initialize_databases():
    """Creates the SQLite databases and tables for each grade if they don't exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    logger.info(f"Ensuring database directory exists at: {os.path.abspath(DB_DIR)}")

    for grade, db_path in GRADE_DBS.items():
        try:
            logger.info(f"Initializing database for Grade {grade} at: {db_path}")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.executescript(DB_SCHEMA)
            conn.commit()
            logger.info(f"Database for Grade {grade} initialized successfully.")
            
            # Verify table creation
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='qa_pairs';")
            if cursor.fetchone():
                logger.debug(f"'qa_pairs' table verified in Grade {grade} database.")
            else:
                logger.error(f"Failed to verify 'qa_pairs' table in Grade {grade} database!")
                
        except sqlite3.Error as e:
            logger.error(f"SQLite error while initializing database for Grade {grade}: {e}")
        finally:
            if conn:
                conn.close()

if __name__ == "__main__":
    logger.info("Starting historical Q&A database initialization...")
    initialize_databases()
    logger.info("Historical Q&A database initialization complete.")
