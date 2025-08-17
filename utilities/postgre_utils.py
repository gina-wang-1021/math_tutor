import sys, pathlib

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from supabase import create_client, Client
from logger_config import setup_logger
from utilities.load_env import load_env_vars

logger = setup_logger(__name__)

url: str = load_env_vars("SUPABASE_URL")
key: str = load_env_vars("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# perhaps add maintenance functions? Clean, check, backup, etc.

def get_historic_answer(id: int, no_extra_explain: bool):
    """Given a known record's ID, see if the correct version of answer exists and fetch the historic answer from the database."""
    
    tablespace_name = "gradeTwelveMath"
    if not id:
        logger.warning("ID is None.")
        return None
    
    answer_field = "ans_high_con" if no_extra_explain else "ans_low_con"
    try:
        query_result = (
            supabase.table(tablespace_name)
            .select(f"id, {answer_field}")
            .eq("id", id)
            .execute()
        )
        
        if not query_result.data:
            logger.warning(f"No record found with id {id} in table {tablespace_name}")
            return None
            
        response = query_result.data[0]
        logger.info(f"Fetched historic answer for id {response['id']}: {response[answer_field]}")
        return response[answer_field]
    except Exception as e:
        logger.error(f"Error getting historic answer: {str(e)}")
        return None

def insert_answer(no_extra_explain: bool, answer: str):

    tablespace_name = "gradeTwelveMath"
    try:
        answer_field = "ans_high_con" if no_extra_explain else "ans_low_con"
        response = (
            supabase.table(tablespace_name)
            .insert(
                {answer_field: answer}
            )
            .execute()
        ).data[0]
        logger.info(f"Inserted answer for id {response['id']}")
        return response['id']
    except Exception as e:
        logger.error(f"Error inserting answer: {str(e)}")
        return None

def update_answer(id: int, no_extra_explain: bool, answer: str):
    tablespace_name = "gradeTwelveMath"
    if not id:
        logger.warning("ID is None.")
        return None
    try:
        answer_field = "ans_high_con" if no_extra_explain else "ans_low_con"
        response = (
            supabase.table(tablespace_name)
            .update({answer_field: answer})
            .eq("id", id)
            .execute()
        ).data[0]
        logger.info(f"Updated answer for id {response['id']}: {answer}")
        return response
    except Exception as e:
        logger.error(f"Error updating answer: {str(e)}")
        return None