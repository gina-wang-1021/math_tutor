import sqlite3
import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings # Corrected import for newer Langchain
from .logger_config import setup_logger
from dotenv import load_dotenv

logger = setup_logger('ingest_mock_historic_data')
load_dotenv()

DB_DIR = "historic_qa_data"
GRADE_DBS = {
    11: os.path.join(DB_DIR, "grade11_historic.db"),
    12: os.path.join(DB_DIR, "grade12_historic.db")
}
FAISS_INDEXES = {
    11: os.path.join(DB_DIR, "grade11.index"),
    12: os.path.join(DB_DIR, "grade12.index")
}

# OpenAI's text-embedding-ada-002 model produces 1536-dimensional embeddings
EMBEDDING_DIM = 1536

MOCK_DATA = {
    11: [
        {"question": "What is the Pythagorean theorem?", 
         "answer": "In a right-angled triangle, the square of the hypotenuse side is equal to the sum of squares of the other two sides (a^2 + b^2 = c^2)."},
        {"question": "Explain quadratic equations.", 
         "answer": "A quadratic equation is a second-order polynomial equation in a single variable x, ax^2 + bx + c = 0, where a != 0."},
        {"question": "What are the properties of a parallelogram?",
         "answer": "Opposite sides are equal and parallel, opposite angles are equal, and diagonals bisect each other."}
    ],
    12: [
        {"question": "What are derivatives in calculus?", 
         "answer": "A derivative represents the rate at which a function is changing at any given point. It measures the slope of the tangent line to the graph of the function at that point."},
        {"question": "Explain the concept of limits.", 
         "answer": "A limit describes the value that a function approaches as the input (or index) approaches some value. Limits are essential to calculus and mathematical analysis."},
        {"question": "What is integration?",
         "answer": "Integration is a fundamental concept in calculus, which, in its simplest form, can be thought of as the reverse process of differentiation, or as a way to sum up infinitesimal parts to find a whole, like an area under a curve."}
    ]
}

def ingest_data():
    """Ingests mock Q&A data into SQLite and creates FAISS indexes."""
    os.makedirs(DB_DIR, exist_ok=True)
    logger.info(f"Ensuring data directory exists at: {os.path.abspath(DB_DIR)}")

    try:
        embeddings_model = OpenAIEmbeddings()
    except Exception as e:
        logger.error(f"Failed to initialize OpenAIEmbeddings. Ensure OPENAI_API_KEY is set. Error: {e}")
        return

    for grade, qa_list in MOCK_DATA.items():
        db_path = GRADE_DBS[grade]
        faiss_index_path = FAISS_INDEXES[grade]
        logger.info(f"Processing Grade {grade} data...")

        question_texts = [qa['question'] for qa in qa_list]
        answer_texts = [qa['answer'] for qa in qa_list]
        
        if not question_texts:
            logger.info(f"No mock data for Grade {grade}. Skipping.")
            continue

        try:
            logger.info(f"Generating embeddings for {len(question_texts)} questions for Grade {grade}...")
            question_vectors = embeddings_model.embed_documents(question_texts)
            question_vectors_np = np.array(question_vectors, dtype='float32')
            logger.info(f"Embeddings generated successfully for Grade {grade}.")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for Grade {grade}: {e}")
            continue

        # Initialize FAISS index
        # IndexIDMap2 allows us to use our own IDs (from SQLite) with the vectors
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(EMBEDDING_DIM))
        
        sqlite_ids = []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            logger.info(f"Connected to SQLite DB: {db_path}")

            for i, qa in enumerate(qa_list):
                try:
                    cursor.execute("INSERT INTO qa_pairs (question_text, answer_text) VALUES (?, ?)", 
                                   (qa['question'], qa['answer']))
                    conn.commit()
                    last_id = cursor.lastrowid
                    sqlite_ids.append(last_id)
                    logger.debug(f"Inserted Q: '{qa['question'][:30]}...' A: '{qa['answer'][:30]}...' with ID: {last_id} into Grade {grade} DB.")
                except sqlite3.IntegrityError: # Handles UNIQUE constraint violation for question_text
                    logger.warning(f"Question already exists, fetching ID: '{qa['question'][:30]}...' for Grade {grade}")
                    cursor.execute("SELECT id FROM qa_pairs WHERE question_text = ?", (qa['question'],))
                    existing_id = cursor.fetchone()
                    if existing_id:
                        sqlite_ids.append(existing_id[0])
                        # If question exists, we might want to update its vector in FAISS or skip adding
                        # For this mock script, we'll assume new vectors for existing IDs are not added to FAISS to keep it simple
                        # Or, ensure mock questions are unique if this path is problematic
                    else:
                        logger.error(f"Could not fetch ID for existing question: {qa['question'][:30]}...")
                        # Decide how to handle this - skip this entry for FAISS?
                        continue # Skip adding this vector to FAISS if ID fetch fails
            
            if len(sqlite_ids) == len(question_vectors_np):
                index.add_with_ids(question_vectors_np, np.array(sqlite_ids, dtype='int64'))
                logger.info(f"Added {index.ntotal} vectors to FAISS index for Grade {grade}.")
                faiss.write_index(index, faiss_index_path)
                logger.info(f"FAISS index for Grade {grade} saved to: {faiss_index_path}")
            else:
                logger.error(f"Mismatch between number of SQLite IDs ({len(sqlite_ids)}) and vectors ({len(question_vectors_np)}) for Grade {grade}. FAISS index not saved.")

        except sqlite3.Error as e:
            logger.error(f"SQLite error for Grade {grade}: {e}")
        except Exception as e:
            logger.error(f"General error during FAISS/SQLite processing for Grade {grade}: {e}")
        finally:
            if 'conn' in locals() and conn:
                conn.close()

if __name__ == "__main__":
    logger.info("Starting mock historical Q&A data ingestion...")
    ingest_data()
    logger.info("Mock historical Q&A data ingestion complete.")
