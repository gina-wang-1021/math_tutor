import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from utilities.prompt_utils import load_prompt
from utilities.load_env import load_env_vars
from logger_config import setup_logger
import openai
import concurrent.futures

logger = setup_logger(__name__)

# Shared index name
INDEX_NAME = "math-tutor"

topic_mapping = {
    "algebra": "algebra",
    "basics of financial mathematics": "financial-math",
    "geometry": "geometry",
    "calculus": "calculus",
    "mathematical reasoning": "reasoning",
    "numbers, quantification and numerical applications": "applications",
    "probability": "probability",
    "statistics": "statistics",
    "set and functions": "sets",
    "surface area and volumes": "areas",
    "trigonometry": "trigonometry"
}

def get_chunks_for_current_year(topic: str, query: str):
    """Get relevant chunks for a topic at twelfth grade level using Pinecone namespaces.
    Note: This function now always fetches chunks for twelfth grade regardless of the year parameter.

    Args:
        topic (str): The topic to search for.
        query (str): The search query.

    Returns:
        list: A list of relevant chunks from twelfth grade level.
    """

    # Always use twelfth grade regardless of the year parameter
    year_str = 'twelve'
    topic = topic_mapping.get(topic, "algebra")
    logger.info(f"Retrieving chunks for topic '{topic}' at level twelfth with query: '{query[:50]}...'")

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
        
        logger.info(f"Retrieved chunks for topic '{topic}'")
        return chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks for topic '{topic}': {str(e)}")
        return []

def get_chunks_from_prior_years(topic: str, query: str):
    """Get relevant chunks for a topic from levels below twelfth grade using Pinecone namespaces.
    Note: This function now always fetches chunks from grades 9, 10, and 11 (excluding grade 12).

    Args:
        topic (str): The topic to search for.
        query (str): The search query.

    Returns:
        list: A list of relevant chunks from grades 9, 10, and 11.
    """
    
    # Always fetch from grades 9, 10, and 11
    prior_levels = ['nine', 'ten', 'eleven']
    topic = topic_mapping.get(topic, "algebra")
    
    logger.info(f"Retrieving chunks for topic '{topic}' with query: '{query[:50]}...')")
    
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
        
        def query_level(level):
            """Query a specific level and return chunks."""
            try:
                results = index.query(
                    vector=query_embedding,
                    top_k=2,
                    namespace=topic,
                    filter={"level": level},
                    include_metadata=True
                )
                
                level_chunks = []
                for match in results.matches:
                    chunk = {
                        'page_content': match.metadata.get('text', ''),
                        'score': match.score
                    }
                    level_chunks.append(chunk)
                return level_chunks
                    
            except Exception as e:
                logger.warning(f"Error querying level '{level}' for topic '{topic}': {str(e)}")
                return []
        
        # Query all levels in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            level_futures = {executor.submit(query_level, level): level for level in prior_levels}
            
            for future in concurrent.futures.as_completed(level_futures):
                level = level_futures[future]
                try:
                    level_chunks = future.result()
                    all_chunks.extend(level_chunks)
                except Exception as e:
                    logger.warning(f"Error getting results for level '{level}': {str(e)}")
                    continue
        
        # Sort by adjusted score and return top 5
        all_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = all_chunks[:5]
        top_chunks = [chunk['page_content'] for chunk in top_chunks]
        
        logger.info(f"Retrieved chunks for topic '{topic}'")
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
            classification = response.lower()
            reasoning = "Parsing failed, using fallback method"
            logger.info(f"Topic classification (fallback): '{classification}' | Question: '{question[:50]}...'")
        
        valid_topics = ["algebra", "basics of financial mathematics", "geometry", "calculus", "mathematical reasoning", "numbers, quantification and numerical applications", "probability", "statistics", "sets and functions", "surface area and volumes", "trigonometry"]
        
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