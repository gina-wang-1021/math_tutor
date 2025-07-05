import os
import faiss
import sqlite3
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from utilities.prompt_utils import load_prompt
from langchain.chains import LLMChain
from logger_config import setup_logger

logger = setup_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def search_historic_qa(question_embedding: np.ndarray, faiss_index, db_conn, k: int, max_distance_threshold: float, no_extra_explain: bool):
    """Searches historic Q&A, returns answer and question ID if a sufficiently similar question is found."""
    try:
        if not faiss_index or faiss_index.ntotal == 0:
            logger.info("Skipping FAISS search as the index is empty or not provided.")
            return None, None
            
        distances, ids = faiss_index.search(question_embedding, k)
        logger.debug(f"FAISS search results: distances={distances}, ids={ids}")

        if ids.size == 0 or ids[0][0] == -1: 
            logger.info("No similar historic question found (FAISS ID -1 or empty results).")
            return None, None

        best_match_id = ids[0][0]
        best_match_distance = distances[0][0]

        if best_match_distance < max_distance_threshold:
            logger.info(f"Found similar historic question with ID {best_match_id}, distance {best_match_distance:.4f} (Threshold: < {max_distance_threshold}).")
            cursor = db_conn.cursor()
            
            # Determine which answer field to retrieve based on the no_extra_explain parameter
            answer_field = "ans_high_con" if no_extra_explain else "ans_low_con"
            logger.info(f"Using {answer_field} based on no_extra_explain={no_extra_explain}")
            
            # First try to get the preferred answer type
            cursor.execute(f"SELECT {answer_field}, id FROM qa_pairs WHERE id = ?", (int(best_match_id),))
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                logger.info(f"Retrieved {answer_field} for ID {best_match_id}.")
                return result[0], result[1]
            else:
                logger.warning(f"Historic answer not found in DB for ID {best_match_id}, though FAISS match found.")
                return None, result[1]
        else:
            logger.info(f"No sufficiently similar historic question found. Closest distance {best_match_distance:.4f} (Threshold: < {max_distance_threshold}).")
            return None, None
    except Exception as e:
        logger.error(f"Error during historic Q&A search: {e}")
        return None, None

def get_historic_answer(grade, question, max_distance_threshold, no_extra_explain):
    """Attempt to find a similar question in the historic Q&A and return its answer."""
    if grade not in [11, 12]:
        return None

    logger.info(f"Attempting to retrieve historic Q&A for grade {grade} for question: '{question[:50]}...'")
    historic_faiss_index = None
    historic_db_conn = None
    try:
        embeddings_model_instance = OpenAIEmbeddings()
        current_question_embedding_list = embeddings_model_instance.embed_query(question)
        current_question_embedding = np.array([current_question_embedding_list], dtype='float32')

        historic_faiss_index, historic_db_conn = load_historic_qa_resources(grade)

        if historic_faiss_index and historic_db_conn:
            historic_answer, historic_answer_id = search_historic_qa(
                current_question_embedding,
                historic_faiss_index,
                historic_db_conn,
                k=1,
                max_distance_threshold=max_distance_threshold,
                no_extra_explain=no_extra_explain
            )
            if historic_answer is not None:
                logger.info(f"Found and using historic answer for grade {grade} question.")
                return historic_answer, historic_answer_id
            elif historic_answer_id:
                logger.info(f"Found historic answer ID for grade {grade} question, but answer not found in DB.")
                return None, historic_answer_id

        else:
            logger.debug(f"Could not load historic Q&A resources for grade {grade}. Proceeding with normal generation.")
    
    except Exception as e:
        logger.error(f"Error during historic Q&A retrieval attempt for grade {grade}: {e}")
    finally:
        if historic_db_conn:
            historic_db_conn.close()
    
    logger.info(f"Did not find a similar question asked for grade {grade}")
    return None, None

def store_qa_pair(grade: int, question: str, answer: str, historic_answer_id=None, no_extra_explain: bool=False) -> bool:
    """Store a new question-answer pair in the FAISS index and SQLite database or update an existing one.
    
    Args:
        grade (int): The grade level (11 or 12)
        question (str): The question to store
        answer (str): The generated answer to store
        historic_answer_id (int, optional): The ID of an existing entry to update. If None, a new entry is created.
        no_extra_explain (bool): Whether to store the high confidence answer (True) or low confidence answer (False)
        
    Returns:
        bool: True if the operation was successful, False otherwise
    """
    if grade not in [11, 12]:
        logger.warning(f"Cannot store Q&A pair for invalid grade {grade}. Must be 11 or 12.")
        return False
        
    faiss_path = FAISS_INDEX_FILES[grade]
    db_path = GRADE_DBS_FILES[grade]
    
    if not os.path.exists(db_path):
        logger.error(f"SQLite DB file not found for grade {grade} at {db_path}")
        return False
        
    db_conn = None

    try:
        # Connect to SQLite database
        db_conn = sqlite3.connect(db_path)
        cursor = db_conn.cursor()
        
        # Determine which answer field to use based on confidence level
        answer_field = "ans_high_con" if no_extra_explain else "ans_low_con"
        logger.info(f"Using {answer_field} based on no_extra_explain={no_extra_explain}")
        
        last_id = None

        if historic_answer_id:
            # Update existing entry
            logger.info(f"Updating existing Q&A pair with ID {historic_answer_id} for grade {grade}")
            cursor.execute(f"UPDATE qa_pairs SET {answer_field} = ? WHERE id = ?", 
                          (answer, historic_answer_id))
            db_conn.commit()
            
            if cursor.rowcount > 0:
                last_id = historic_answer_id
                logger.info(f"Updated {answer_field} for Q&A pair with ID {last_id}")
            else:
                logger.warning(f"No rows updated for ID {historic_answer_id}. Entry may not exist.")
                return False
        else:
            # Insert new entry
            logger.info(f"Inserting new Q&A pair for grade {grade}: '{question[:50]}...'")
            cursor.execute(f"INSERT INTO qa_pairs (question_text, {answer_field}) VALUES (?, ?)", 
                          (question, answer))
            db_conn.commit()
            last_id = cursor.lastrowid
            logger.info(f"Inserted new Q&A pair with ID {last_id} into SQLite database for grade {grade}")
            
            # Generate embedding for the question (only needed for new entries)
            embeddings_model = OpenAIEmbeddings()
            question_vector = embeddings_model.embed_query(question)
            question_vector_np = np.array([question_vector], dtype='float32')
        
        # Load and update FAISS index if this is a new entry (not an update)
        try:
            # Only update FAISS index for new entries, not for updates to existing entries
            if historic_answer_id is None:
                if os.path.exists(faiss_path):
                    faiss_index = faiss.read_index(faiss_path)
                    logger.info(f"Loaded existing FAISS index for grade {grade} with {faiss_index.ntotal} entries")
                else:
                    logger.warning(f"FAISS index not found for grade {grade}, creating new one")
                    dimension = len(question_vector)
                    base_index = faiss.IndexFlatL2(dimension)
                    faiss_index = faiss.IndexIDMap(base_index)
                    logger.info(f"Created new FAISS index for grade {grade} with dimension {dimension}")
                    
                    # Double check the directory exists
                    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
                
                # Add the new question vector to the FAISS index
                faiss_index.add_with_ids(question_vector_np, np.array([last_id], dtype='int64'))
                logger.info(f"Added vector to FAISS index for grade {grade}, new size: {faiss_index.ntotal}")
                
                # Save the updated FAISS index
                faiss.write_index(faiss_index, faiss_path)
                logger.info(f"Saved updated FAISS index for grade {grade}")
            else:
                # For updates, we don't need to modify the FAISS index as the question hasn't changed
                logger.info(f"Skipping FAISS index update for existing entry ID {historic_answer_id}")

            
            return True
        except Exception as e:
            logger.error(f"Error updating FAISS index for grade {grade}: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error storing Q&A pair for grade {grade}: {e}")
        return False
    finally:
        if db_conn:
            db_conn.close()

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
        llm_chain = rephrase_prompt | llm_rephrase
        response = llm_chain.invoke({"user_question": user_question, "history": history})
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error rephrasing question: {str(e)}")
        return user_question
