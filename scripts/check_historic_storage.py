"""
Script to check the content of FAISS vector stores and SQLite databases
to verify that Q&A pairs are being stored correctly.
"""

import os
import sys
import sqlite3
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
import argparse
from dotenv import load_dotenv

# Add the project root to the path so we can import from utilities
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utilities.qa_utils import FAISS_INDEX_FILES, GRADE_DBS_FILES
from logger_config import setup_logger

logger = setup_logger(__name__)

# Load environment variables from .env file
load_dotenv()

def check_sqlite_database(grade):
    """Check the contents of the SQLite database for a specific grade."""
    if grade not in GRADE_DBS_FILES:
        print(f"Error: Grade {grade} is not supported. Use 11 or 12.")
        return
        
    db_path = GRADE_DBS_FILES[grade]
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='qa_pairs'")
        schema = cursor.fetchone()
        print(f"\nTable Schema for grade {grade}:")
        print(schema[0])
        
        # Count total records
        cursor.execute("SELECT COUNT(*) FROM qa_pairs")
        count = cursor.fetchone()[0]
        print(f"\nTotal Q&A pairs in database: {count}")
        
        # Get the most recent entries
        cursor.execute("SELECT id, question_text, answer_text FROM qa_pairs ORDER BY id DESC LIMIT 5")
        recent_entries = cursor.fetchall()
        
        print(f"\nMost recent entries (up to 5):")
        for entry in recent_entries:
            print(f"ID: {entry[0]}")
            print(f"Question: {entry[1][:100]}..." if len(entry[1]) > 100 else f"Question: {entry[1]}")
            print(f"Answer: {entry[2][:100]}..." if len(entry[2]) > 100 else f"Answer: {entry[2]}")
            print("-" * 50)
            
        conn.close()
        
    except Exception as e:
        print(f"Error accessing SQLite database: {e}")

def check_faiss_index(grade, query=None):
    """Check the FAISS index for a specific grade and optionally search for a query."""
    if grade not in FAISS_INDEX_FILES:
        print(f"Error: Grade {grade} is not supported. Use 11 or 12.")
        return
        
    faiss_path = FAISS_INDEX_FILES[grade]
    if not os.path.exists(faiss_path):
        print(f"Error: FAISS index file not found at {faiss_path}")
        return
        
    try:
        # Load the FAISS index
        index = faiss.read_index(faiss_path)
        print(f"\nFAISS index for grade {grade}:")
        print(f"Total vectors: {index.ntotal}")
        print(f"Vector dimension: {index.d}")
        
        # If a query is provided, search for similar questions
        if query:
            print(f"\nSearching for: '{query}'")
            embeddings_model = OpenAIEmbeddings()
            query_vector = embeddings_model.embed_query(query)
            query_vector_np = np.array([query_vector], dtype='float32')
            
            # Search the index
            k = 3  # Number of nearest neighbors to retrieve
            distances, ids = index.search(query_vector_np, k)
            
            print(f"\nTop {k} similar questions:")
            if ids.size > 0 and ids[0][0] != -1:
                # Connect to the database to get the actual questions
                db_path = GRADE_DBS_FILES[grade]
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                for i in range(min(k, len(ids[0]))):
                    if ids[0][i] != -1:  # Valid ID
                        cursor.execute("SELECT question_text FROM qa_pairs WHERE id = ?", (int(ids[0][i]),))
                        result = cursor.fetchone()
                        if result:
                            similarity = 1.0 - min(distances[0][i] / 2.0, 1.0)  # Convert distance to similarity score
                            print(f"ID: {ids[0][i]}, Similarity: {similarity:.4f}")
                            print(f"Question: {result[0][:100]}..." if len(result[0]) > 100 else f"Question: {result[0]}")
                            print("-" * 50)
                conn.close()
            else:
                print("No similar questions found.")
                
    except Exception as e:
        print(f"Error accessing FAISS index: {e}")

def main():
    parser = argparse.ArgumentParser(description='Check historic Q&A storage')
    parser.add_argument('--grade', type=int, choices=[11, 12], required=True, help='Grade level to check (11 or 12)')
    parser.add_argument('--query', type=str, help='Optional query to search for similar questions')
    parser.add_argument('--db-only', action='store_true', help='Check only the SQLite database')
    parser.add_argument('--faiss-only', action='store_true', help='Check only the FAISS index')
    
    args = parser.parse_args()
    
    print(f"Checking historic Q&A storage for grade {args.grade}...\n")
    
    if not args.faiss_only:
        check_sqlite_database(args.grade)
        
    if not args.db_only:
        check_faiss_index(args.grade, args.query)

if __name__ == "__main__":
    main()
