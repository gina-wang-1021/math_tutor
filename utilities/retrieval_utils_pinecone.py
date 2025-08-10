import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from utilities.prompt_utils import load_prompt
from utilities.load_env import load_env_vars
from logger_config import setup_logger
import openai

logger = setup_logger(__name__)

# Shared index name
INDEX_NAME = "math-tutor"

topic_mapping = {
    "algebra": "algebra",
    "basics of financial mathematics": "financial-math",
    "geometry": "geometry",
    "calculus": "calculus",
    "mathematical reasoning": "reasoning",
    "numbers quantification and numerical applications": "applications",
    "probability": "probability",
    "statistics": "statistics",
    "set and functions": "sets",
    "surface area and volumes": "areas",
    "trigonometry": "trigonometry"
}

def get_chunks_for_current_year(topic: str, year: int, query: str):
    """Get relevant chunks for a topic at the specified year level using Pinecone namespaces.

    Args:
        topic (str): The topic to search for.
        year (int): The year level (11 or 12).
        query (str): The search query.

    Returns:
        list: A list of relevant chunks from the specified level.
    """

    int_str_year_mapping = {
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve"
    }
    year_str = int_str_year_mapping[year]
    topic = topic_mapping.get(topic, "algebra")
    logger.info(f"Retrieving chunks for topic '{topic}' at level '{year_str}' with query: '{query[:50]}...'")
    
    level_order = ['nine', 'ten', 'eleven', 'twelve']
    if year_str not in level_order:
        logger.warning(f"Invalid level '{year_str}', defaulting to 'twelve'.")
        year_str = 'twelve'

    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=load_env_vars("PINECONE_API_KEY"))
        
        # Check if index exists
        if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
            logger.error(f"No Pinecone index found: '{INDEX_NAME}'")
            return []

        # Get the index
        index = pc.Index(INDEX_NAME)
        
        # Create embedding for query
        openai.api_key = load_env_vars("OPENAI_API_KEY")
        embedding_response = openai.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Query the specific namespace (topic) with level filter
        results = index.query(
            vector=query_embedding,
            top_k=3,
            namespace=topic,
            filter={"level": year_str},
            include_metadata=True
        )
        
        chunks = []
        for match in results.matches:
            # Create a document-like object with the content and metadata
            chunk = match.metadata.get('text', '')
            chunks.append(chunk)
        
        logger.info(f"Retrieved {len(chunks)} chunks for topic '{topic}' at level '{year_str}'")
        return chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks for topic '{topic}' at level '{year_str}': {str(e)}")
        return []

def get_chunks_from_prior_years(topic: str, year: int, query: str):
    """Get relevant chunks for a topic from levels below the specified year using Pinecone namespaces.

    Args:
        topic (str): The topic to search for.
        year (int): The current year level (9, 10, 11, 12).
        query (str): The search query.

    Returns:
        list: A combined list of relevant chunks from all prior levels.
    """
    
    level_order = ['nine', 'ten', 'eleven', 'twelve']
    int_str_year_mapping = {
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve"
    }
    year_str = int_str_year_mapping.get(year, 'twelve')
    
    current_level_index = level_order.index(year_str)
    prior_levels = level_order[:current_level_index]
    topic = topic_mapping.get(topic, "algebra")
    
    logger.info(f"Retrieving chunks from levels {prior_levels} for topic '{topic}' with query: '{query[:50]}...'")
    
    try:
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=load_env_vars("PINECONE_API_KEY"))
        
        # Check if index exists
        if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
            logger.error(f"No Pinecone index found: '{INDEX_NAME}'")
            return []

        # Get the index
        index = pc.Index(INDEX_NAME)
        
        # Create embedding for query
        openai.api_key = load_env_vars("OPENAI_API_KEY")
        embedding_response = openai.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = embedding_response.data[0].embedding
        
        all_chunks = []
        
        # Query each level with different k values (more for higher levels)
        for level in prior_levels:
            k_value = 3 if level in ['eleven', 'twelve'] else 2
            
            try:
                results = index.query(
                    vector=query_embedding,
                    top_k=k_value,
                    namespace=topic,
                    filter={"level": level},
                    include_metadata=True
                )
                
                for match in results.matches:
                    # Create a document-like object with the content and metadata
                    chunk = {
                        'page_content': match.metadata.get('text', ''),
                        'score': match.score + (level_order.index(level) * 0.1)
                    }
                    all_chunks.append(chunk)
                    
            except Exception as e:
                logger.warning(f"Error querying level '{level}' for topic '{topic}': {str(e)}")
                continue
        
        # Sort by adjusted score and return top 5
        all_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = all_chunks[:5]
        top_chunks = [chunk['page_content'] for chunk in top_chunks]
        
        logger.info(f"Retrieved {len(top_chunks)} total chunks for topic '{topic}' from levels {prior_levels}")
        return top_chunks

    except Exception as e:
        logger.error(f"Error retrieving prior-year chunks for topic '{topic}': {str(e)}")
        return []

def get_topic(question: str) -> str:
    """Determine the math topic based on the question. 
    Returns:
        - The most relevant topic
    """
    try:
        llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.0)
        topic_prompt_text = load_prompt("topic_classification_prompt.txt")
        topic_prompt = PromptTemplate.from_template(topic_prompt_text)
        
        topic_chain = topic_prompt | llm
        response = topic_chain.invoke({"question": question}).content.strip()
        
        # Parse the Python list response
        try:
            import ast
            parsed_response = ast.literal_eval(response)
            if isinstance(parsed_response, list) and len(parsed_response) == 2:
                classification = parsed_response[0].lower() if (parsed_response[0] != "None" or parsed_response[0] != "none") else "none"
                reasoning = parsed_response[1]
                logger.info(f"Topic classification: '{classification}' | Reasoning: {reasoning}")
            else:
                raise ValueError("Response is not a valid list with 2 elements")
        except (ValueError, SyntaxError) as parse_error:
            logger.warning(f"Failed to parse response as list: '{response}'. Error: {parse_error}")
            # Fallback to old parsing method
            classification = response.lower()
            reasoning = "Parsing failed, using fallback method"
            logger.info(f"Topic classification (fallback): '{classification}' | Question: '{question[:50]}...'")
        
        valid_topics = ["algebra", "basics of financial mathematics", "geometry", "calculus", "mathematical reasoning", "numbers quantification and numerical applications", "probability", "statistics", "set and functions", "surface area and volumes", "trigonometry"]
        
        if classification in valid_topics:
            return classification

        if classification == "none" or classification == "None":
            logger.info("Question not covered in our topics, not answering the question.")
            return "none"
        
        logger.warning(f"Topic detection returned an unexpected response: '{classification}'. Defaulting to 'algebra' for question: '{question[:50]}...' ")
        return "algebra"
        
    except Exception as e:
        logger.error(f"Error detecting topics for question '{question[:50]}...': {str(e)}")
        return "algebra"

if __name__ == "__main__":
    answer = get_topic("A solution of 8% is to be diluted by adding a 2% boric acid solution to it. The resulting mixture is to be more than 4% but less than  6% borid acid. If we have 640 litres of the 8% solution, how many litres of the 2% solution will have to be added?")
    print(answer)