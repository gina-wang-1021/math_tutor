from pinecone import Pinecone, ServerlessSpec
import os
import openai
import sys, pathlib
# Ensure project root is on sys.path so we can import load_env when this file is run directly
project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from load_env import load_env_vars
load_env_vars()

openai.api_key = os.environ["OPENAI_API_KEY"]

index_id = 0

def create_index():
    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
    )
    pc.create_index(
        name="math-tutor",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

def embed(docs: list[str]) -> list[list[float]]:
    res = openai.embeddings.create(
        input=docs,
        model="text-embedding-3-small"
    )
    doc_embeds = [r.embedding for r in res.data] 
    return doc_embeds 

def add_to_index(docs: list[str]):
    x = embed(docs)
    vectors = []
    for vec in x:
        # TODO: switch this to base on the last inserted SQLite database entry
        global index_id
        vectors.append({
            "id": str(index_id),
            "values": vec
        })
        index_id += 1

    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
    )
    index = pc.Index("math-tutor")
    index.upsert(
        vectors=vectors,
        namespace="math-tutor"
    )
    print("Added to index")
    return
        

def search_index(query: str):
    try:
        pc = Pinecone(
            api_key=os.environ["PINECONE_API_KEY"],
        )
        index = pc.Index("math-tutor")

        x = embed([query])

        results = index.query(
            namespace="math-tutor",
            vector=x[0],
            top_k=1,
            include_values=False,
            include_metadata=False
        )["matches"][0]
    except Exception as e:
        logger.error(f"Error searching index: {str(e)}")
        return None

    if float(results["score"]) >= 0.95:
        return results["id"]
    else:
        return None

if __name__ == "__main__":
    # add_to_index(["What is 2+2?"])
    print(bool(search_index("What is 2+2?")))
    print(bool(search_index("how do I write code")))