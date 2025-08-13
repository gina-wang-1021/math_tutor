from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from logger_config import setup_logger
from utilities.student_utils import get_confidence_level_and_score, check_confidence_and_score
from utilities.prompt_utils import load_prompt
from utilities.qa_utils import rephrase_question, fetch_historic_data, store_new_data
from utilities.retrieval_utils_pinecone import get_chunks_for_current_year, get_chunks_from_prior_years, get_topic
import time
import ast
import concurrent.futures

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self, stream_handler, delay=0.05):
        """Initialize with a streaming handler function.
        
        Args:
            stream_handler (callable): Function that handles each token
            delay (float): Time in seconds to delay between tokens (default: 0.05)
        """
        self.stream_handler = stream_handler
        self.delay = delay
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM produces a new token."""
        if self.stream_handler:
            self.stream_handler(token)
            time.sleep(self.delay)

# Initialize logger
logger = setup_logger('engine')

# Test all log levels (can be kept here or removed if not essential for engine.py's direct operation)
logger.debug('Test DEBUG message')
logger.info('Test INFO message')
logger.warning('Test WARNING message')
logger.error('Test ERROR message')

def pipeline(student_data, user_question, history, stream_handler=None):
    """Process a question and return an answer.
    
    Args:
        student_data (dict): The student's data
        user_question (str): The question to answer
        history (str): Chat history
        stream_handler (callable, optional): Function to handle streaming tokens
            The function should accept a string token as its argument.
    """
    try:
        student_id = student_data["Student ID"]
        
        logger.info(f"Processing question for student {student_id}")
        logger.debug(f"Original question: {user_question}")
        logger.debug(f"History length: {len(history.split()) if history else 0} words")
        # Rephrase the question with context
        try:
            t0 = time.perf_counter()
            logger.info("Rephrasing question...")
            standalone_question = rephrase_question(user_question, history)
            logger.info(f"Rephrased Question → {standalone_question}")
            logger.info(f"Rephrasing question took {time.perf_counter() - t0:.2f} sec")
            
            # Detect topics from the rephrased question
            t1 = time.perf_counter()
            logger.info("Detecting topic...")
            detected_topic = get_topic(standalone_question)
            logger.info(f"Get topic result: {detected_topic}")
            if detected_topic == "none":
                response_message = f"This question is not covered in your textbook yet. I can only answer math questions related to your textbook content - happy to help with those! 😊"

                if stream_handler:
                    tokens = response_message.split()
                    for token in tokens:
                        stream_handler(token + " ")
                        time.sleep(0.05)
                    return ""
                else:
                    return response_message
            logger.info(f"Detecting topic took {time.perf_counter() - t1:.2f} sec")

            # Initialize LLMs - one for intermediate steps (no streaming) and one for final answer (with streaming)
            llm_intermediate = ChatOpenAI(
                model_name="gpt-4.1",
                temperature=0.5,
                streaming=False
            )
            
            llm_final = ChatOpenAI(
                model_name="gpt-4.1",
                temperature=0.5,
                streaming=bool(stream_handler),
                callbacks=[StreamingCallbackHandler(stream_handler, delay=0.05)] if stream_handler else None
            )
            
            # Get the confidence level and scores for the topic
            logger.info("Getting student's confidence level and score for topic...")
            topic_level, topic_scores = get_confidence_level_and_score(student_data, detected_topic)
            logger.debug(f"Student confidence level and score for topic {detected_topic}: {topic_level}, {topic_scores}")
            
            # Check confidence level and topic scores
            no_extra_explain = check_confidence_and_score(topic_level, topic_scores)
            logger.info(f"confidence and score check result: {no_extra_explain}")
            
            # Detect whether a similar question had been asked before
            t4 = time.perf_counter()
            historic_answer, historic_answer_id = fetch_historic_data(standalone_question, no_extra_explain)
            if historic_answer:
                if stream_handler:
                    logger.info(f"Streaming historic answer.")
                    for _, char in enumerate(historic_answer):
                        stream_handler(char)
                        time.sleep(0.02)
                    logger.info(f"Detecting whether a similar question had been asked before took {time.perf_counter() - t4:.2f} sec")
                    return ""
                else:
                    logger.info(f"Detecting whether a similar question had been asked before took {time.perf_counter() - t4:.2f} sec")
                    return historic_answer
            logger.info(f"Detecting whether a similar question had been asked before took {time.perf_counter() - t4:.2f} sec")

            # Retrieve chunks in parallel
            t5 = time.perf_counter()
            logger.info("Retrieving chunks in parallel...")
            
            current_year_chunks = []
            lower_years_chunk = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both chunk retrieval tasks
                current_year_future = executor.submit(get_chunks_for_current_year, detected_topic, standalone_question)
                prior_years_future = executor.submit(get_chunks_from_prior_years, detected_topic, standalone_question)
                
                # Topic-based answer prompt
                topic_based_answer_prompt_text = load_prompt("topic_based_answer_prompt.txt")
                topic_prompt = PromptTemplate.from_template(topic_based_answer_prompt_text)
                
                # Get results from both tasks
                try:
                    current_year_chunks = current_year_future.result()
                    if current_year_chunks:
                        logger.debug(f"Found {len(current_year_chunks)} relevant chunks for {detected_topic}")
                    else:
                        logger.warning(f"No relevant chunks found for {detected_topic} at level twelfth")
                except Exception as e:
                    logger.error(f"Error getting chunks for {detected_topic}: {str(e)}")
                    current_year_chunks = []
                
                try:
                    lower_years_chunk = prior_years_future.result()
                    if lower_years_chunk:
                        logger.debug(f"Found {len(lower_years_chunk)} relevant chunks for {detected_topic}")
                    else:
                        logger.warning(f"No relevant chunks found for {detected_topic} below level twelfth")
                except Exception as e:
                    logger.error(f"Error getting chunks from prior years for {detected_topic}: {str(e)}")
                    lower_years_chunk = []

            # Process chunks (now they are text strings, not Document objects)
            processed_current_year_chunks = "\n\n".join(current_year_chunks) if current_year_chunks else ""
            processed_lower_years_chunks = "\n\n".join(lower_years_chunk) if lower_years_chunk else ""
            logger.info(f"Retrieve chunks took {time.perf_counter() - t5:.2f} sec")

            # Check if we should use a single pass based on the previously calculated confidence check
            if no_extra_explain:
                try:
                    # Use streaming LLM to generate direct answer
                    t6 = time.perf_counter()
                    answer_chain = topic_prompt | llm_final
                    logger.info("Generating single pass answer...")

                    final_answer = answer_chain.invoke({
                        "retrieved_chunks": processed_current_year_chunks,
                        "topic": detected_topic,
                        "student_question": standalone_question,
                    }).content.strip()
                    logger.info(f"Final Answer for topic-based question → {final_answer}")
                    logger.info(f"Generating single pass answer took {time.perf_counter() - t6:.2f} sec")

                    t11 = time.perf_counter()
                    store_success = store_new_data(standalone_question, final_answer, no_extra_explain, historic_answer_id)
                    if store_success:
                        logger.info(f"Successfully stored Q&A pair (single pass)")
                    else:
                        logger.warning(f"Failed to store Q&A pair (single pass)")
                    logger.info(f"Storing Q&A pair (single pass) took {time.perf_counter() - t11:.2f} sec")
                    return final_answer
                except Exception as e:
                    logger.error(f"Error generating answer: {str(e)}")
                    return "I'm sorry, I encountered an error while generating an answer. Please try again."
            
            # first pass
            try:
                logger.info("Generating first pass answer...")
                t7 = time.perf_counter()
                first_pass_chain = topic_prompt | llm_intermediate
                first_response = first_pass_chain.invoke({
                    "retrieved_chunks": processed_current_year_chunks,
                    "topic": detected_topic,
                    "student_question": standalone_question
                }).content.strip()
                logger.info(f"First Pass Answer for topic-based question → {first_response}")
                logger.info(f"Generating first pass answer took {time.perf_counter() - t7:.2f} sec")
            except Exception as e:
                logger.error(f"Error generating first pass answer: {str(e)}")
                return "I'm sorry, I encountered an error while generating an answer. Please try again."

            # Second pass
            logger.info("Generating second pass answer...")
            t8 = time.perf_counter()
            
            logger.info("Generating detail explanation for topic-based question...")
            explain_prompt_text = load_prompt("explain_prompt.txt")
            explain_prompt = PromptTemplate.from_template(explain_prompt_text)
            answer_chain = explain_prompt | llm_intermediate

            second_response = answer_chain.invoke({
                "retrieved_chunks": processed_lower_years_chunks,
                "first_pass_answer": first_response,
                "student_question": standalone_question
            }).content.strip()
            logger.info(f"Second Pass Answer for topic-based question → {second_response}")
            logger.info(f"Generating second pass answer took {time.perf_counter() - t8:.2f} sec")

            logger.info("Comparing answers...")
            t9 = time.perf_counter()
            compare_prompt_text = load_prompt("compare_prompt.txt")
            compare_prompt = PromptTemplate.from_template(compare_prompt_text)
            compare_chain = compare_prompt | llm_final
            
            compare_answer = compare_chain.invoke({
                "first_pass_answer": first_response,
                "second_pass_answer": second_response,
                "student_question": standalone_question
            }).content.strip()
            logger.info(f"Compared answer: {compare_answer}")
            logger.info(f"Comparing answers took {time.perf_counter() - t9:.2f} sec")
            
            # Save generated answer and the student's question to both databases
            t10 = time.perf_counter()
            store_success = store_new_data(standalone_question, compare_answer, no_extra_explain, historic_answer_id)
            if store_success:
                logger.info(f"Successfully stored Q&A pair (two-pass)")
            else:
                logger.warning(f"Failed to store Q&A pair (two-pass)")
            logger.info(f"Storing Q&A pair took {time.perf_counter() - t10:.2f} sec")
            return compare_answer

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return "I'm having trouble accessing the learning materials. Please try again later."
            
    except Exception as e:
        logger.error(f"Error processing your question: {str(e)}")
        return "I'm having trouble understanding your question. Could you please rephrase it?"
        
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {str(e)}")
        return "I'm sorry, something went wrong. Our team has been notified. Please try again later."