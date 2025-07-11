import os
import sys, pathlib

# Add project root to sys.path before importing local modules
project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from supabase import create_client, Client
from logger_config import setup_logger
from utilities.load_env import load_env_vars

logger = setup_logger(__name__)
load_env_vars()

url: str = os.environ["SUPABASE_URL"]
key: str = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

# perhaps add maintenance functions? Clean, check, backup, etc.

def get_historic_answer(id: int, no_extra_explain: bool, tablespace_name: str):
    """Given a known record's ID, see if the correct version of answer exists and fetch the historic answer from the database."""
    answer_field = "ans_high_con" if no_extra_explain else "ans_low_con"
    try:
        response = (
            supabase.table(tablespace_name)
            .select(f"id, {answer_field}")
            .eq("id", id)
            .execute()
        ).data[0]
        logger.info(f"Fetched historic answer for id {response['id']}: {response[answer_field]}")
        return response[answer_field]
    except Exception as e:
        logger.error(f"Error getting historic answer: {str(e)}")
        return None

def insert_answer(no_extra_explain: bool, answer: str, tablespace_name: str):
    try:
        answer_field = "ans_high_con" if no_extra_explain else "ans_low_con"
        response = (
            supabase.table(tablespace_name)
            .insert(
                {answer_field: answer}
            )
            .execute()
        ).data[0]
        print(response)
        logger.info(f"Inserted answer for id {response['id']}: {answer}")
        return response['id']
    except Exception as e:
        logger.error(f"Error inserting answer: {str(e)}")
        return None

def update_answer(id: int, no_extra_explain: bool, answer: str, tablespace_name: str):
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

if __name__ == "__main__":
    insert_answer(True, "1+1 is 2", "gradeElevenMath")