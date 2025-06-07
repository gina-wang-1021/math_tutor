# one time script for creating RAG indices

import os
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

def build_index_for_folder(base_folder, topic, persist_path):
    all_docs = []
    
    topic_folder = os.path.join(base_folder, topic)
    for filename in os.listdir(topic_folder):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}")
            print(f"Level: {file_level_mapping[filename[:4]]}")
            user_input = input("Is this correct? (y/n)")
            if user_input == "y":
                level = file_level_mapping[filename[:4]]
                loader = PyPDFLoader(os.path.join(topic_folder, filename))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["level"] = level
                    doc.metadata["topic"] = topic
            else:
                break

            all_docs.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)

    load_dotenv()

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )
    db.persist()
    print(f"Saved Chroma index to {persist_path}")
    
    
build_index_for_folder("sorted_docs", "geometry", "indexes/geometry")

