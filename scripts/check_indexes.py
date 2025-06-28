import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Check if OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set it in your .env file or directly in your environment.")
    sys.exit(1)

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Updated path to indexes directory
INDEXES_DIR = "/Users/wangyichi/LocalData/chromadb/indexes"

def check_index(topic):
    """Check if an index exists and contains data for a given topic."""
    index_path = os.path.join(INDEXES_DIR, topic)
    
    if not os.path.exists(index_path):
        print(f"❌ No index directory found for topic '{topic}'")
        return False
    
    try:
        # Try to load the Chroma DB
        db = Chroma(persist_directory=index_path, embedding_function=OpenAIEmbeddings())
        
        # Check if there are any documents in the collection
        collection = db._collection
        count = collection.count()
        
        if count > 0:
            print(f"✅ Topic '{topic}' has {count} documents in the index")
            
            # Get the available levels
            levels = set()
            docs = collection.get()
            if docs and 'metadatas' in docs and docs['metadatas']:
                for metadata in docs['metadatas']:
                    if metadata and 'level' in metadata:
                        levels.add(metadata['level'])
            
            if levels:
                print(f"   Available levels: {', '.join(sorted(levels))}")
            return True
        else:
            print(f"❌ Topic '{topic}' has an index but it contains 0 documents")
            return False
    
    except Exception as e:
        print(f"❌ Error checking index for topic '{topic}': {str(e)}")
        return False

def main():
    # Get all topic directories
    indexes_dir = INDEXES_DIR
    if not os.path.exists(indexes_dir):
        print(f"❌ Indexes directory not found at {indexes_dir}")
        return
    
    topics = [d for d in os.listdir(indexes_dir) 
              if os.path.isdir(os.path.join(indexes_dir, d)) and not d.startswith('.')]
    
    if not topics:
        print("❌ No topic directories found in the indexes folder")
        return
    
    print(f"Found {len(topics)} topic directories: {', '.join(topics)}")
    print("\nChecking each index for data:")
    print("=" * 50)
    
    valid_topics = []
    for topic in topics:
        if check_index(topic):
            valid_topics.append(topic)
    
    print("=" * 50)
    print(f"\nSummary: {len(valid_topics)}/{len(topics)} topics have valid indexes with data")
    
    if valid_topics:
        print("\nTest query on a valid index:")
        try:
            test_topic = valid_topics[0]
            index_path = os.path.join(INDEXES_DIR, test_topic)
            db = Chroma(persist_directory=index_path, embedding_function=OpenAIEmbeddings())
            
            # Try a simple query
            test_query = "What is this topic about?"
            results = db.similarity_search(test_query, k=1)
            
            if results:
                print(f"\n✅ Successfully queried the '{test_topic}' index")
                print(f"Sample result: {results[0].page_content[:100]}...")
            else:
                print(f"\n❌ Query returned no results for '{test_topic}'")
        
        except Exception as e:
            print(f"\n❌ Error testing query: {str(e)}")

if __name__ == "__main__":
    main()
