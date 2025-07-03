import os
import shutil
import chromadb
import logging
import datetime
import time
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Set up logging
log_dir = "/Users/wangyichi/Documents/Projects/math_tutor/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"build_index_{datetime.datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('build_index')

file_level_mapping = {
    "iemh": "nine",
    "jemh": "ten",
    "kemh": "eleven",
    "lemh": "twelve"
}

def build_index_for_folder(base_folder, topic):
    """Build Chroma index for a specific topic folder
    
    Args:
        base_folder (str): Base directory containing topic subdirectories
        topic (str): Name of the topic subdirectory to process
    """
    all_docs = []
    topic_folder = os.path.join(base_folder, topic)
    persist_path = os.path.join(indexes_folder, topic)
    
    # Clear existing index if it exists
    if os.path.exists(persist_path):
        logger.info(f"Removing existing index at {persist_path}")
        shutil.rmtree(persist_path)
    
    logger.info(f"Processing topic: {topic}")
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

    logger.info(f"Creating embeddings and Chroma index...")
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )
    db.persist()
    logger.info(f"Saved Chroma index to {persist_path}")
    time.sleep(1)


if __name__ == "__main__":
    load_dotenv()
    # Use the new paths in LocalData/chromadb
    base_folder = "/Users/wangyichi/LocalData/chromadb/sorted_docs"
    indexes_folder = "/Users/wangyichi/LocalData/chromadb/indexes"
    
    # Create indexes directory if it doesn't exist
    os.makedirs(indexes_folder, exist_ok=True)
    
    # Process each topic subdirectory
    for topic in os.listdir(base_folder):
        topic_path = os.path.join(base_folder, topic)
        if os.path.isdir(topic_path):
            try:
                build_index_for_folder(base_folder, topic)
                logger.info(f"Successfully built index for topic {topic}")
            except Exception as e:
                logger.error(f"Error processing topic {topic}: {str(e)}")
    
    logger.info("Index building complete!")

