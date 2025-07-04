import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from utilities.prompt_utils import load_prompt
from langchain.chains import LLMChain
from logger_config import setup_logger

logger = setup_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to ChromaDB indexes
INDEXES_DIR = "/Users/wangyichi/LocalData/chromadb/indexes"

def get_chunks_for_current_year(topic, year, query):
    """Get relevant chunks for a topic at the specified year level.

    Args:
        topic (str): The topic to search for.
        year (str): The year level ('nine', 'ten', 'eleven', 'twelve').
        query (str): The search query.

    Returns:
        list: A list of relevant chunks from the specified level.
    """
    logger.info(f"Retrieving chunks for topic '{topic}' at level '{year}' with query: '{query[:50]}...'")
    
    level_order = ['nine', 'ten', 'eleven', 'twelve']
    if year not in level_order:
        logger.warning(f"Invalid level '{year}', defaulting to 'twelve'.")
        year = 'twelve'

    try:
        topic = topic.replace(" ", "_")
        index_path = os.path.join(INDEXES_DIR, topic)
        if not os.path.exists(index_path):
            logger.error(f"No index found for topic '{topic}' at '{index_path}'.")
            return []

        db = Chroma(persist_directory=index_path, embedding_function=OpenAIEmbeddings())
        
        chunks = db.similarity_search(query, k=3, filter={"level": year})
        
        if chunks:
            logger.debug(f"Found {len(chunks)} chunks for level '{year}' in topic '{topic}'.")
            for chunk in chunks:
                if 'score' not in chunk.metadata:
                    chunk.metadata['score'] = 0.0
        else:
            logger.debug(f"No chunks found for level '{year}' in topic '{topic}'.")
            
        return chunks

    except Exception as e:
        logger.error(f"Error retrieving chunks for topic '{topic}' at level '{year}': {e}")
        return []

def get_chunks_from_prior_years(topic, year, query):
    """Get relevant chunks for a topic from levels below the specified year.

    Args:
        topic (str): The topic to search for.
        year (str): The current year level ('nine', 'ten', 'eleven', 'twelve').
        query (str): The search query.

    Returns:
        list: A combined list of relevant chunks from all prior levels.
    """
    logger.info(f"Retrieving prior-year chunks for topic '{topic}' up to (but not including) level '{year}'.")

    level_order = ['nine', 'ten', 'eleven', 'twelve']
    if year not in level_order:
        logger.warning(f"Invalid level '{year}', no prior levels to fetch.")
        return []

    max_level_idx = level_order.index(year)
    if max_level_idx == 0:
        logger.info(f"No prior levels to retrieve for the lowest level '{year}'.")
        return []

    try:
        topic = topic.replace(" ", "_")
        index_path = os.path.join(INDEXES_DIR, topic)
        if not os.path.exists(index_path):
            logger.error(f"No index found for topic '{topic}' at '{index_path}'.")
            return []

        db = Chroma(persist_directory=index_path, embedding_function=OpenAIEmbeddings())
        all_chunks = []

        for current_level in level_order[:max_level_idx]:
            try:
                chunks = db.similarity_search(query, k=3, filter={"level": current_level})
                if chunks:
                    logger.debug(f"Found {len(chunks)} chunks for level '{current_level}' in topic '{topic}'.")
                    for chunk in chunks:
                        if 'score' not in chunk.metadata:
                            chunk.metadata['score'] = 0.0
                    all_chunks.extend(chunks)
                else:
                    logger.debug(f"No chunks found for level '{current_level}' in topic '{topic}'.")
            except Exception as e:
                logger.warning(f"Error retrieving chunks for topic '{topic}', level '{current_level}': {e}")
                continue
        
        logger.info(f"Retrieved a total of {len(all_chunks)} chunks from prior years for topic '{topic}'.")
        return all_chunks

    except Exception as e:
        logger.error(f"Error retrieving prior-year chunks for topic '{topic}': {e}")
        return []

def get_topic(question):
    """Determine the math topic based on the question. 
    Returns:
        - None if it's a calculation question
        - 'overview' if it's asking about general capabilities
        - The most relevant topic otherwise
    """
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        topic_prompt_text = load_prompt("topic_classification_prompt.txt")
        topic_prompt = PromptTemplate.from_template(topic_prompt_text)
        
        topic_chain = topic_prompt | llm
        response = topic_chain.invoke({"question": question}).content.strip().lower()
        
        if response == "calculation":
            logger.info(f"Question classified as 'calculation': '{question[:50]}...' ")
            return None
        elif response == "overview":
            logger.info(f"Question classified as 'overview': '{question[:50]}...' ")
            return "overview"
        
        valid_topics = ["algebra", "basics of financial mathematics", "geometry", "calculus", "mathematical reasoning", "numbers quantification and numerical applications", "probability", "statistics", "set and functions", "surface area and volumes", "trigonometry"]
        if response in valid_topics:
            logger.info(f"Question classified under topic '{response}': '{question[:50]}...' ")
            return response
        
        logger.warning(f"Topic detection returned an unexpected response: '{response}'. Defaulting to 'algebra' for question: '{question[:50]}...' ")
        return "algebra"
        
    except Exception as e:
        logger.error(f"Error detecting topics for question '{question[:50]}...': {str(e)}")
        return "algebra"
