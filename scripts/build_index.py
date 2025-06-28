# Script for creating RAG indices from PDF files in sorted_docs directory

import os
import shutil
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

file_level_mapping = {
    "iemh": "beginner",
    "jemh": "intermediate",
    "kemh": "advanced",
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
        print(f"Removing existing index at {persist_path}")
        shutil.rmtree(persist_path)
    
    print(f"\nProcessing topic: {topic}")
    for filename in os.listdir(topic_folder):
        if not filename.endswith(".pdf"):
            continue
            
        file_prefix = filename[:4]
        if file_prefix not in file_level_mapping:
            print(f"Warning: Skipping {filename} - unknown level prefix")
            continue
            
        level = file_level_mapping[file_prefix]
        print(f"Processing {filename} (Level: {level})")
        
        try:
            loader = PyPDFLoader(os.path.join(topic_folder, filename))
            docs = loader.load()
            for doc in docs:
                doc.metadata["level"] = level
                doc.metadata["topic"] = topic
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    if not all_docs:
        print(f"No documents processed for topic {topic}")
        return
        
    print(f"Splitting {len(all_docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)

    print(f"Creating embeddings and Chroma index...")
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )
    db.persist()
    print(f"Saved Chroma index to {persist_path}")


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
                print(f"Successfully built index for topic {topic}")
            except Exception as e:
                print(f"Error processing topic {topic}: {str(e)}")
    
    print("\nIndex building complete!")

