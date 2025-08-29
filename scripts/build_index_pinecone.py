import os
import sys
import logging
import datetime
import time
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import openai

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utilities.load_env import load_env_vars

# Set up logging
log_dir = "/Users/wangyichi/Documents/Projects/math_tutor/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"build_index_pinecone_{datetime.datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('build_index_pinecone')

file_level_mapping = {
    "iemh": "nine",
    "jemh": "ten",
    "kemh": "eleven",
    "lemh": "twelve"
}

math_topic_mapping = {
    "algebra": "algebra",
    "basics_of_financial_mathematics": "financial-math",
    "geometry": "geometry",
    "calculus": "calculus",
    "mathematical_reasoning": "reasoning",
    "numbers_quantification_and_numerical_applications": "applications",
    "probability": "probability",
    "statistics": "statistics",
    "set_and_functions": "sets",
    "surface_area_and_volumes": "areas",
    "trigonometry": "trigonometry"
}

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create embeddings for a list of texts using OpenAI API (following your existing pattern)."""
    res = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    embeddings = [r.embedding for r in res.data]
    logger.info(f"Generated embeddings for {len(texts)} texts")
    return embeddings

def build_math_index_for_folder(base_folder, topic, topic_record, pc, index_name):
    """Build Pinecone index for a specific math topic folder using namespaces
    
    Args:
        base_folder (str): Base directory containing topic subdirectories
        topic (str): Name of the topic subdirectory to process
        pc (Pinecone): Pinecone client instance
        index_name (str): Name of the single Pinecone index to use
    """

    all_docs = []
    topic_folder = os.path.join(base_folder, topic)
    
    logger.info(f"Processing topic: {topic_record}")
    for filename in os.listdir(topic_folder):
        if not filename.endswith(".txt"):
            continue
            
        file_prefix = filename[:4]
        if file_prefix not in file_level_mapping:
            logger.warning(f"Skipping {filename} - unknown level prefix")
            continue
            
        level = file_level_mapping[file_prefix]
        logger.info(f"Processing {filename} (Level: {level})")
        time.sleep(1)
        
        try:
            loader = TextLoader(os.path.join(topic_folder, filename))
            docs = loader.load()
            for doc in docs:
                doc.metadata["level"] = level
                doc.metadata["topic"] = topic_record
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    if not all_docs:
        logger.warning(f"No documents processed for topic {topic_record}")
        return
        
    logger.info(f"Splitting {len(all_docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
    print(chunks[0].metadata)

    # Upsert to namespace in the shared index
    try:
        # Create embeddings and upsert to Pinecone using namespace
        logger.info(f"Creating embeddings and upserting to namespace '{topic_record}'...")
        
        # Extract text content and create embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings_list = embed_texts(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
            # Create clean metadata without source field
            clean_metadata = {
                "level": chunk.metadata.get("level"),
                "topic": chunk.metadata.get("topic"),
                "text": chunk.page_content
            }
            
            vectors.append({
                "id": f"{topic_record}_{i}",
                "values": embedding,
                "metadata": clean_metadata
            })
        
        # Upsert to Pinecone using namespace
        index = pc.Index(index_name)
        batch_size = 50
        successful_batches = 0
        total_batches = (len(vectors) - 1) // batch_size + 1
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                index.upsert(vectors=batch, namespace=topic_record)
                successful_batches += 1
                logger.info(f"Uploaded batch {i//batch_size + 1}/{total_batches} ({len(batch)} vectors) to namespace '{topic_record}'")
            except Exception as batch_error:
                logger.error(f"Error upserting batch {i//batch_size + 1} to namespace '{topic_record}': {str(batch_error)}")
                # Continue with next batch instead of failing completely
                continue
        
        if successful_batches > 0:
            logger.info(f"Successfully uploaded {successful_batches}/{total_batches} batches to namespace '{topic_record}' in index {index_name}")
        else:
            logger.error(f"Failed to upload any batches to namespace '{topic_record}' in index {index_name}")
            raise Exception(f"All batch uploads failed for namespace '{topic_record}'")
        
    except Exception as e:
        logger.error(f"Error upserting to namespace '{topic_record}' in index {index_name}: {str(e)}")
        raise

def build_econ_index_for_folder(base_folder, topic, pc, index_name):
    """Build Pinecone index for a specific economics topic folder using namespaces
    
    Args:
        base_folder (str): Base directory containing topic subdirectories
        topic (str): Name of the topic subdirectory to process
        pc (Pinecone): Pinecone client instance
        index_name (str): Name of the single Pinecone index to use
    """

    all_docs = []
    topic_folder = os.path.join(base_folder, topic)
    
    logger.info(f"Processing topic: {topic}")
    for filename in os.listdir(topic_folder):
        if not filename.endswith(".txt"):
            continue
        
        try:
            loader = TextLoader(os.path.join(topic_folder, filename))
            docs = loader.load()
            for doc in docs:
                doc.metadata["topic"] = topic
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    if not all_docs:
        logger.warning(f"No documents processed for topic {topic}")
        return
        
    logger.info(f"Splitting {len(all_docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
    print(chunks[0].metadata)

    # Upsert to namespace in the shared index
    try:
        # Create embeddings and upsert to Pinecone using namespace
        logger.info(f"Creating embeddings and upserting to namespace '{topic}'...")
        
        # Extract text content and create embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings_list = embed_texts(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
            # Create clean metadata without source field
            clean_metadata = {
                "topic": topic,
                "text": chunk.page_content
            }
            
            vectors.append({
                "id": f"{topic}_{i}",
                "values": embedding,
                "metadata": clean_metadata
            })
        
        # Upsert to Pinecone using namespace
        index = pc.Index(index_name)
        batch_size = 50
        successful_batches = 0
        total_batches = (len(vectors) - 1) // batch_size + 1
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                index.upsert(vectors=batch, namespace=topic)
                successful_batches += 1
                logger.info(f"Uploaded batch {i//batch_size + 1}/{total_batches} ({len(batch)} vectors) to namespace '{topic}'")
            except Exception as batch_error:
                logger.error(f"Error upserting batch {i//batch_size + 1} to namespace '{topic}': {str(batch_error)}")
                # Continue with next batch instead of failing completely
                continue
        
        if successful_batches > 0:
            logger.info(f"Successfully uploaded {successful_batches}/{total_batches} batches to namespace '{topic}' in index {index_name}")
        else:
            logger.error(f"Failed to upload any batches to namespace '{topic}' in index {index_name}")
            raise Exception(f"All batch uploads failed for namespace '{topic}'")
        
    except Exception as e:
        logger.error(f"Error upserting to namespace '{topic}' in index {index_name}: {str(e)}")
        raise

def math_index_execute():
    if not load_env_vars():
        logger.warning(".env file not found. Using system environment variables.")
    
    # Initialize Pinecone and OpenAI
    if not os.environ.get("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY environment variable is not set")
        return
        
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        return
        
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    # Use the same paths as ChromaDB version
    base_folder = "/Users/wangyichi/LocalData/chromadb/sorted_docs"
    
    # Create single shared index name
    index_name = "math-tutor"
    
    # Create or get existing shared index
    try:
        if index_name not in [index.name for index in pc.list_indexes()]:
            logger.info(f"Creating new shared Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            time.sleep(10)  # Wait for index to be ready
        else:
            logger.info(f"Using existing shared Pinecone index: {index_name}")
    except Exception as e:
        logger.error(f"Error creating/accessing shared index {index_name}: {str(e)}")
        return
    
    # Process each topic subdirectory as a namespace
    for topic in os.listdir(base_folder):
        topic_record = topic_mapping.get(topic, "algebra")
        topic_path = os.path.join(base_folder, topic)
        if os.path.isdir(topic_path):
            try:
                build_math_index_for_folder(base_folder, topic, topic_record, pc, index_name)
                logger.info(f"Successfully built namespace '{topic_record}' in shared index")
                
                # Wait for user input before processing next folder
                input(f"\nFinished processing '{topic_record}'. Press Enter to continue to next folder...")
                
            except Exception as e:
                logger.error(f"Error processing topic {topic_record}: {str(e)}")
                
                # Also wait on error so user can see the error message
                input(f"\nError occurred with '{topic_record}'. Press Enter to continue to next folder...")
    
    logger.info("Pinecone namespace building complete!")

def econ_index_execute():
    if not load_env_vars("PINECONE_API_KEY") or not load_env_vars("OPENAI_API_KEY"):
        logger.warning(".env file not found. Using system environment variables.")
        
    pc = Pinecone(api_key=load_env_vars("PINECONE_API_KEY"))
    openai.api_key = load_env_vars("OPENAI_API_KEY")
    
    # Use the same paths as ChromaDB version
    base_folder = "/Users/wangyichi/LocalData/chromadb/econ_sorted_docs"
    
    # Create single shared index name
    index_name = "econ-tutor"
    
    # Create or get existing shared index
    try:
        if index_name not in [index.name for index in pc.list_indexes()]:
            logger.info(f"Creating new shared Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            time.sleep(10)
        else:
            logger.info(f"Using existing shared Pinecone index: {index_name}")
    except Exception as e:
        logger.error(f"Error creating/accessing shared index {index_name}: {str(e)}")
        return
    
    # Process each topic subdirectory as a namespace
    for topic in os.listdir(base_folder):
        topic_path = os.path.join(base_folder, topic)
        if os.path.isdir(topic_path):
            try:
                build_econ_index_for_folder(base_folder, topic, pc, index_name)
                logger.info(f"Successfully built namespace '{topic}' in shared index")
                
            except Exception as e:
                logger.error(f"Error processing topic {topic}: {str(e)}")
                input(f"\nError occurred with '{topic}'. Press Enter to continue to next folder...")
    
    logger.info("Pinecone namespace building complete!")

if __name__ == "__main__":
    econ_index_execute()
