import os
import faiss
import sqlite3
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from logger_config import setup_logger

logger = setup_logger(__name__)

# Path to the prompts directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(PROJECT_ROOT, "prompts")

def load_prompt(prompt_name: str) -> str:
    """Loads a prompt string from a file in the PROMPT_DIR."""
    prompt_file_path = os.path.join(PROMPT_DIR, prompt_name)
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt file {prompt_file_path}: {e}")
        raise

# Determine the project root (PROJECT_ROOT is defined above with PROMPT_DIR)
HISTORIC_QA_DIR_NAME = "historic_qa_data"
HISTORIC_QA_DIR = os.path.join(PROJECT_ROOT, HISTORIC_QA_DIR_NAME)

FAISS_INDEX_FILES = {
    11: os.path.join(HISTORIC_QA_DIR, "grade11.index"),
    12: os.path.join(HISTORIC_QA_DIR, "grade12.index")
}
GRADE_DBS_FILES = {
    11: os.path.join(HISTORIC_QA_DIR, "grade11_historic.db"),
    12: os.path.join(HISTORIC_QA_DIR, "grade12_historic.db")
}
MAX_L2_DISTANCE_THRESHOLD = 0.2 
HISTORIC_QA_K_NEIGHBORS = 1

def load_historic_qa_resources(grade: int):
    """Loads FAISS index and SQLite DB connection for the given grade."""
    if grade not in FAISS_INDEX_FILES:
        logger.warning(f"No historic Q&A configured for grade {grade}. Known grades: {list(FAISS_INDEX_FILES.keys())}")
        return None, None

    faiss_path = FAISS_INDEX_FILES[grade]
    db_path = GRADE_DBS_FILES[grade]
    loaded_faiss_index = None
    db_conn = None

    if not os.path.exists(faiss_path):
        logger.warning(f"FAISS index file not found for grade {grade} at {faiss_path}")
        return None, None
    if not os.path.exists(db_path):
        logger.warning(f"SQLite DB file not found for grade {grade} at {db_path}")
        return None, None

    try:
        loaded_faiss_index = faiss.read_index(faiss_path)
        if loaded_faiss_index.ntotal == 0:
            logger.warning(f"FAISS index for grade {grade} at {faiss_path} is empty.")
        logger.info(f"FAISS index loaded for grade {grade} from {faiss_path}. Index size: {loaded_faiss_index.ntotal}")
    except Exception as e:
        logger.error(f"Failed to load FAISS index for grade {grade} from {faiss_path}: {e}")
        return None, None

    try:
        db_conn = sqlite3.connect(db_path)
        logger.info(f"SQLite DB connection established for grade {grade} at {db_path}")
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite DB for grade {grade} at {db_path}: {e}")
        return None, None
    
    return loaded_faiss_index, db_conn

def search_historic_qa(question_embedding: np.ndarray, faiss_index, db_conn, k: int, max_distance_threshold: float):
    """Searches historic Q&A, returns answer if a sufficiently similar question is found."""
    try:
        if not faiss_index or faiss_index.ntotal == 0:
            logger.info("Skipping FAISS search as the index is empty or not provided.")
            return None
            
        distances, ids = faiss_index.search(question_embedding, k)
        logger.debug(f"FAISS search results: distances={distances}, ids={ids}")

        if ids.size == 0 or ids[0][0] == -1: 
            logger.info("No similar historic question found (FAISS ID -1 or empty results).")
            return None

        best_match_id = ids[0][0]
        best_match_distance = distances[0][0]

        if best_match_distance < max_distance_threshold:
            logger.info(f"Found similar historic question with ID {best_match_id}, distance {best_match_distance:.4f} (Threshold: < {max_distance_threshold}).")
            cursor = db_conn.cursor()
            cursor.execute("SELECT answer_text FROM qa_pairs WHERE id = ?", (int(best_match_id),))
            result = cursor.fetchone()
            if result:
                logger.info(f"Retrieved historic answer for ID {best_match_id}.")
                return result[0]
            else:
                logger.warning(f"Historic answer not found in DB for ID {best_match_id}, though FAISS match found.")
                return None
        else:
            logger.info(f"No sufficiently similar historic question found. Closest distance {best_match_distance:.4f} (Threshold: < {max_distance_threshold}).")
            return None
    except Exception as e:
        logger.error(f"Error during historic Q&A search: {e}")
        return None

def rephrase_question(user_question: str, history: str) -> str:
    """Generate a standalone version of a follow-up question using chat history.

    Args:
        user_question (str): The raw question from the student.
        history (str): Concatenated chat history.

    Returns:
        str: Reformulated standalone question (falls back to original on error).
    """
    try:
        llm_rephrase = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        rephrase_prompt_text = load_prompt("rephrase_question_prompt.txt")
        rephrase_prompt = PromptTemplate.from_template(rephrase_prompt_text)
        llm_chain = LLMChain(llm=llm_rephrase, prompt=rephrase_prompt)
        return llm_chain.run({"user_question": user_question, "history": history}).strip()
    except Exception as e:
        logger.error(f"Error rephrasing question: {str(e)}")
        return user_question
