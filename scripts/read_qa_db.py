import sqlite3
import os

# Path to the database
db_path = '/Users/wangyichi/Documents/Projects/math_tutor/historic_qa_data/grade12_historic.db'

# Check if the database file exists
if not os.path.exists(db_path):
    print(f"Database file not found at {db_path}")
    exit(1)

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get table schema
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='qa_pairs'")
schema = cursor.fetchone()
print("Table Schema:")
print(schema[0] if schema else "No schema found")
print("\n" + "-"*50 + "\n")

# Count total records
cursor.execute("SELECT COUNT(*) FROM qa_pairs")
count = cursor.fetchone()
print(f"Total Q&A pairs in database: {count[0] if count else 0}")
print("\n" + "-"*50 + "\n")

# Get all records
cursor.execute("SELECT id, question_text, answer_text FROM qa_pairs")
rows = cursor.fetchall()

if rows:
    print(f"Found {len(rows)} question-answer pairs:\n")
    for row in rows:
        print(f"ID: {row[0]}")
        print(f"Question: {row[1]}")
        print(f"Answer: {row[2][:200]}..." if len(row[2]) > 200 else f"Answer: {row[2]}")
        print("-"*50)
else:
    print("No question-answer pairs found in the database.")

# Close the connection
conn.close()
