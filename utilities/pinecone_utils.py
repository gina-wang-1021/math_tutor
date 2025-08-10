from pinecone import Pinecone
import openai
import sys, pathlib

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utilities.load_env import load_env_vars
from logger_config import setup_logger

logger = setup_logger(__name__)

openai.api_key = load_env_vars("OPENAI_API_KEY")

def embed(docs: list[str]) -> list[list[float]]:
    res = openai.embeddings.create(
        input=docs,
        model="text-embedding-3-small"
    )
    doc_embeds = [r.embedding for r in res.data] 
    logger.info(f"Embedded generated response")
    return doc_embeds 

def add_to_index(docs: list[str], id: int):
    database_name = "grade-twelve-math"

    x = embed(docs)
    vectors = []
    for vec in x:
        vectors.append({
            "id": str(id),
            "values": vec
        })

    pc = Pinecone(
        api_key=load_env_vars("PINECONE_API_KEY"),
    )
    index = pc.Index(database_name)
    index.upsert(
        vectors=vectors
    )
    logger.info(f"Stored generated response to index with id {id}")
    return

def search_index(query: str):

    database_name = "grade-twelve-math" 

    try:
        pc = Pinecone(
            api_key=load_env_vars("PINECONE_API_KEY"),
        )
        index = pc.Index(database_name)

        x = embed([query])

        query_results = index.query(
            vector=x[0],
            top_k=1,
            include_values=False,
            include_metadata=False
        )
        
        matches = query_results.get("matches", [])
        if not matches:
            logger.info(f"No matches found for query: {query} in {database_name}")
            return None
            
        results = matches[0]
    except Exception as e:
        logger.error(f"Error searching index: {str(e)}")
        return None

    if float(results["score"]) >= 0.85:
        logger.info(f"Found historic answer for query: {query} with id: {results['id']} in {database_name} and score: {round(results['score'], 4)}")
        return results["id"]
    else:
        logger.info(f"No historic answer found for query: {query} with score: {round(results['score'], 4)} in {database_name}")
        return None