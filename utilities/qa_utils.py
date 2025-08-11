import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from utilities.prompt_utils import load_prompt
from logger_config import setup_logger
from utilities.postgre_utils import insert_answer, update_answer, get_historic_answer
from utilities.pinecone_utils import search_index, add_to_index

logger = setup_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def rephrase_question(user_question: str, history: str) -> str:
    """Generate a standalone version of a follow-up question using chat history.

    Args:
        user_question (str): The raw question from the student.
        history (str): Concatenated chat history.

    Returns:
        str: Reformulated standalone question (falls back to original on error).
    """
    try:
        llm_rephrase = ChatOpenAI(model_name="gpt-4.1", temperature=0.0)
        rephrase_prompt_text = load_prompt("rephrase_question_prompt.txt")
        rephrase_prompt = PromptTemplate.from_template(rephrase_prompt_text)
        llm_chain = rephrase_prompt | llm_rephrase
        response = llm_chain.invoke({"user_question": user_question, "history": history})
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error rephrasing question: {str(e)}")
        return user_question

def fetch_historic_data(query: str, no_extra_explain: bool):
    answer_index = search_index(query)
    if not answer_index:
        return None, None
    historic_answer = get_historic_answer(int(answer_index), no_extra_explain)
    if not historic_answer:
        return None, int(answer_index)
    return historic_answer, int(answer_index)

def store_new_data(query: str, answer: str, no_extra_explain: bool, question_id=None):    
    if question_id:
        update_answer(question_id, no_extra_explain, answer)
        return True
    try:
        insert_position = insert_answer(no_extra_explain, answer)
        if insert_position is None:
            logger.error("Failed to insert answer to database, skipping vector storage")
            return False
        add_to_index(query, insert_position)
        return True
    except Exception as e:
        logger.error(f"Error inserting answer: {str(e)}")
        return False