import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
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
INDEXES_DIR_NAME = "indexes"

def get_relevant_chunks(topic, year, query):
    """Get relevant chunks for a topic up to and including the specified year level.
    
    Args:
        topic (str): The topic to search for
        year (str): The maximum year level ('beginner', 'intermediate', 'advanced')
        query (str): The search query
        
    Returns:
        list: Combined list of relevant chunks from all applicable levels
    """
    logger.info(f"Retrieving chunks for topic {topic} up to level {year} using query: '{query[:50]}...' ")
    
    level_order = ['beginner', 'intermediate', 'advanced']
    if year not in level_order:
        logger.warning(f"Invalid level {year}, defaulting to beginner")
        year = 'beginner'
    max_level_idx = level_order.index(year)
    
    try:
        index_path = os.path.join(PROJECT_ROOT, INDEXES_DIR_NAME, topic)
        if not os.path.exists(index_path):
            logger.error(f"No index found for topic {topic} at {index_path}")
            available_indexes_dir = os.path.join(PROJECT_ROOT, INDEXES_DIR_NAME)
            logger.debug(f"Available indexes in '{available_indexes_dir}': {os.listdir(available_indexes_dir) if os.path.exists(available_indexes_dir) else 'directory not found'}")
            return []
            
        db = Chroma(persist_directory=index_path, embedding_function=OpenAIEmbeddings())
        
        all_chunks = []
        
        for current_processing_level in level_order[:max_level_idx + 1]:
            # Get more results from the student's actual current level, fewer from prior levels.
            k_value = 3 if current_processing_level == year else 2 
            
            try:
                chunks = db.similarity_search(
                    query,
                    k=k_value,
                    filter={"level": current_processing_level}
                )
                
                if chunks:
                    logger.debug(f"Found {len(chunks)} chunks for level {current_processing_level} in topic {topic}")
                    for chunk in chunks:
                        if 'score' not in chunk.metadata: # Chroma might not add score by default
                            chunk.metadata['score'] = 0.0 # Placeholder if not present
                    all_chunks.extend(chunks)
                else:
                    logger.debug(f"No chunks found for level {current_processing_level} in topic {topic}")
                    
            except Exception as e:
                logger.warning(f"Error retrieving chunks for topic {topic}, level {current_processing_level}: {str(e)}")
                continue
        
        logger.info(f"Retrieved total of {len(all_chunks)} chunks for topic {topic} up to level {year}")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks for topic {topic}: {str(e)}")
        return []

def get_topic(question):
    """Determine the math topics based on the question. 
    Returns:
        - None if it's a calculation question
        - 'overview' if it's asking about general capabilities
        - The most relevant topic otherwise
    """
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        topic_prompt_text = load_prompt("topic_classification_prompt.txt")
        topic_prompt = PromptTemplate.from_template(topic_prompt_text)
        
        topic_chain = LLMChain(llm=llm, prompt=topic_prompt)
        response = topic_chain.run({"question": question}).strip().lower()
        
        if response == "calculation":
            logger.info(f"Question classified as 'calculation': '{question[:50]}...' ")
            return None
        elif response == "overview":
            logger.info(f"Question classified as 'overview': '{question[:50]}...' ")
            return "overview"
        
        valid_topics = ["algebra", "basics", "geometry", "miscellaneous", "modelling", "probability", "statistics"]
        if response in valid_topics:
            logger.info(f"Question classified under topic '{response}': '{question[:50]}...' ")
            return response
        
        logger.warning(f"Topic detection returned an unexpected response: '{response}'. Defaulting to 'basics' for question: '{question[:50]}...' ")
        return "basics"
        
    except Exception as e:
        logger.error(f"Error detecting topics for question '{question[:50]}...': {str(e)}")
        return "basics"
