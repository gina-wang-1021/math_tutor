from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Removed: from langchain_chroma import Chroma (now in retrieval_utils)
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
# Removed: import pandas as pd (now in student_utils)
import os
from logger_config import setup_logger
# Removed: import faiss (now in qa_utils)
# Removed: import sqlite3 (now in qa_utils)
import numpy as np

# Import from new utility modules
from utilities.student_utils import get_student_level, get_student_topic_level
from utilities.qa_utils import (
    load_historic_qa_resources, 
    search_historic_qa, 
    rephrase_question,
    MAX_L2_DISTANCE_THRESHOLD, # Constant used by pipeline
    HISTORIC_QA_K_NEIGHBORS # Constant used by pipeline
)
from utilities.retrieval_utils import get_relevant_chunks, get_topic


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self, stream_handler):
        """Initialize with a streaming handler function.
        
        Args:
            stream_handler (callable): Function that handles each token
        """
        self.stream_handler = stream_handler
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM produces a new token."""
        if self.stream_handler:
            self.stream_handler(token)

# Initialize logger
logger = setup_logger('engine')

# Path to the prompts directory
# Assuming engine.py is in the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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

# Constants for Historic Q&A are now in utilities.qa_utils
# MAX_L2_DISTANCE_THRESHOLD and HISTORIC_QA_K_NEIGHBORS are imported from qa_utils

# Test all log levels (can be kept here or removed if not essential for engine.py's direct operation)
logger.debug('Test DEBUG message')
logger.info('Test INFO message')
logger.warning('Test WARNING message')
logger.error('Test ERROR message')

# All helper functions (get_student_level, load_historic_qa_resources, 
# search_historic_qa, get_student_topic_level, get_relevant_chunks, 
# get_topic, rephrase_question) have been moved to their respective utility files.

def pipeline(student_id, user_question, history, stream_handler=None, historic_qa_l2_threshold=MAX_L2_DISTANCE_THRESHOLD):
    """Process a question and return an answer.
    
    Args:
        student_id (str): The student's ID
        user_question (str): The question to answer
        history (str): Chat history
        stream_handler (callable, optional): Function to handle streaming tokens
            The function should accept a string token as its argument.
    """
    try:
        logger.info(f"Processing question for student {student_id}")
        logger.debug(f"Original question: {user_question}")
        logger.debug(f"History length: {len(history.split()) if history else 0} words")

        # Rephrase the question with context
        try:
            standalone_question = rephrase_question(user_question, history)
            logger.info(f"Rephrased Question → {standalone_question}")
            
            # Detect topics from the rephrased question
            detected_topic = get_topic(standalone_question)
            logger.info(f"Detected topic: {detected_topic if detected_topic else 'calculation only'}")

            # Initialize LLM for answering
            llm_answer = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.5,
                streaming=bool(stream_handler),
                callbacks=[StreamingCallbackHandler(stream_handler)] if stream_handler else None
            )

            # Handle different types of questions
            if not detected_topic:
                # It's a calculation question
                calculation_prompt_text = load_prompt("calculation_answer_prompt.txt")
                calculation_prompt = PromptTemplate.from_template(calculation_prompt_text)
                
                answer_chain = LLMChain(llm=llm_answer, prompt=calculation_prompt)
                response = answer_chain.run({"question": standalone_question}).strip()
                logger.info(f"Final Answer for calculation question → {response}")
                return response
            
            elif detected_topic == "overview":
                # It's a general inquiry about capabilities
                overview_prompt_text = load_prompt("overview_answer_prompt.txt")
                overview_prompt = PromptTemplate.from_template(overview_prompt_text)
                
                answer_chain = LLMChain(llm=llm_answer, prompt=overview_prompt)
                response = answer_chain.run({"question": standalone_question}).strip()
                logger.info(f"Final Answer for overview question → {response}")
                return response

            # Determine student's overall grade-based level (mapped from grade 11/12)
            grade_based_level, numeric_grade = get_student_level(student_id)
            if grade_based_level is None and numeric_grade is None: # student_id not found or other error in get_student_level
                grade_based_level = "beginner" # Default level string
                # numeric_grade remains None, historic Q&A will be skipped.
                logger.info(f"Student {student_id}: Grade Based Level: {grade_based_level}, Numeric Grade: {numeric_grade}")

            # Detect whether a similar question had been asked before
            if numeric_grade in [11, 12]:
                logger.info(f"Attempting to retrieve historic Q&A for grade {numeric_grade} for question: '{standalone_question[:50]}...'")
                historic_faiss_index = None # Ensure it's defined for finally block
                historic_db_conn = None   # Ensure it's defined for finally block
                try:
                    # Initialize embeddings model here if not already available globally or passed in.
                    # This assumes OPENAI_API_KEY is set in the environment.
                    embeddings_model_instance = OpenAIEmbeddings()
                    current_question_embedding_list = embeddings_model_instance.embed_query(standalone_question)
                    # FAISS expects a 2D array for searching (batch of 1 embedding)
                    current_question_embedding = np.array([current_question_embedding_list], dtype='float32')

                    historic_faiss_index, historic_db_conn = load_historic_qa_resources(numeric_grade)

                    if historic_faiss_index and historic_db_conn:
                        historic_answer = search_historic_qa(
                            current_question_embedding,
                            historic_faiss_index,
                            historic_db_conn,
                            k=HISTORIC_QA_K_NEIGHBORS,
                            max_distance_threshold=historic_qa_l2_threshold
                        )
                        if historic_answer:
                            logger.info(f"Found and using historic answer for grade {numeric_grade} question.")
                            return historic_answer
                    else:
                        logger.debug(f"Could not load historic Q&A resources for grade {numeric_grade}. Proceeding with normal generation.")
                
                except Exception as e:
                    logger.error(f"Error during historic Q&A retrieval attempt for grade {numeric_grade}: {e}")
                finally:
                    if historic_db_conn:
                        historic_db_conn.close()

            logger.info(f"Did not find a similar question asked for grade {numeric_grade}")

            # For topic-based questions, first get all student levels
            topic_level = get_student_topic_level(student_id, detected_topic)
            student_level_info = f"- {detected_topic}: {topic_level}"
            logger.debug(f"Student level for topic {detected_topic}: {topic_level}")
            
            # Retrieve relevant documents for each topic up to the student's level
            all_docs = []
            try:
                docs = get_relevant_chunks(detected_topic, grade_based_level, standalone_question)
                if docs:
                    logger.debug(f"Found {len(docs)} relevant chunks for {detected_topic}")
                    all_docs.extend(docs)
                else:
                    logger.warning(f"No relevant chunks found for {detected_topic} up to level {grade_based_level}")
            except Exception as e:
                logger.error(f"Error getting chunks for {detected_topic}: {str(e)}")
            
            # Sort documents by score
            selected_docs = sorted(
                all_docs,
                key=lambda x: float(x.metadata.get('score', 0)),
                reverse=True
            )[:5]  # Get top 5 chunks
            
            # TODO:
            # implement similarity score threshold
            # if no chunks are above the threshold, return "This topic is not covered in your textbook yet."

            logger.debug(f"Selected {len(selected_docs)} most relevant chunks")
            
            chunks = []
            for doc in selected_docs:
                level = doc.metadata.get('level', 'beginner')
                chunks.append(f"[{level.title()}] {doc.page_content}")
                
            chunks = "\n\n".join(chunks)

            # Determine if student's question is covered in the retrieved chunks
            chunk_coverage_prompt_text = load_prompt("chunk_coverage_prompt.txt")
            chunk_coverage_prompt = PromptTemplate.from_template(chunk_coverage_prompt_text)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
            coverage_answer_chain = LLMChain(llm=llm, prompt=chunk_coverage_prompt)
            coverage_answer = coverage_answer_chain.run({
                "student_question": standalone_question,
                "retrieved_chunks": chunks
            }).strip()
            logger.debug(f"Coverage answer: {coverage_answer}")

            if coverage_answer.strip() == "No" or coverage_answer.strip() == "no":
                return f"This topic is not covered in your textbook yet. I'm happy to help with other questions you have! 😊"
            
            # Topic-based answer prompt
            topic_based_answer_prompt_text = load_prompt("topic_based_answer_prompt.txt")
            topic_prompt = PromptTemplate.from_template(topic_based_answer_prompt_text)

            # Generate answer
            try:
                answer_chain = LLMChain(llm=llm_answer, prompt=topic_prompt)
                logger.info("Generating answer for topic-based question")

                response = answer_chain.run({
                    "retrieved_chunks": chunks,
                    "topic": detected_topic,
                    "student_question": standalone_question,
                    "confidence_level": student_level_info
                })
                logger.info(f"Final Answer for topic-based question → {response}")

                # TODO:
                # Save generated answer and the student's question to both databases

                return response
                
            except Exception as e:
                logger.error(f"Error generating answer: {str(e)}")
                return "I'm sorry, I encountered an error while generating an answer. Please try again."
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return "I'm having trouble accessing the learning materials. Please try again later."
            
    except Exception as e:
        logger.error(f"Error processing your question: {str(e)}")
        return "I'm having trouble understanding your question. Could you please rephrase it?"
        
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {str(e)}")
        return "I'm sorry, something went wrong. Our team has been notified. Please try again later."

if __name__ == "__main__":
    pipeline("beginner", "basics", "what is 10+10", "")