from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from logger_config import setup_logger
from utilities.student_utils import get_student_level, get_confidence_level_and_score, check_confidence_and_score
from utilities.prompt_utils import load_prompt
from utilities.qa_utils import rephrase_question, fetch_historic_data, store_new_data
from utilities.retrieval_utils_pinecone import get_chunks_for_current_year, get_chunks_from_prior_years, get_topic
import time
import ast

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
            logger.info("Rephrasing question...")
            standalone_question = rephrase_question(user_question, history)
            logger.info(f"Rephrased Question → {standalone_question}")
            
            # Detect topics from the rephrased question
            logger.info("Detecting topic...")
            detected_topic = get_topic(standalone_question)
            logger.info(f"Detected topic: {detected_topic}")

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

            # Determine student's overall grade and corresponding database
            logger.info("Determining student's school year...")
            vectorstore_name, tablespace_name, numeric_grade = get_student_level(student_data)
            if not vectorstore_name and not tablespace_name and not numeric_grade:
                vectorstore_name = "grade_eleven_math"
                tablespace_name = "gradeElevenMath"
                numeric_grade = 11
            logger.info(f"Student {student_id}: Vector Database: {vectorstore_name}, Tablespace: {tablespace_name}, Numeric Grade: {numeric_grade}")
            
            # Get the confidence level and scores for the topic
            logger.info("Getting student's confidence level and score for topic...")
            topic_level, topic_scores = get_confidence_level_and_score(student_data, detected_topic)
            logger.debug(f"Student confidence level and score for topic {detected_topic}: {topic_level}, {topic_scores}")
            
            # Check confidence level and topic scores
            no_extra_explain = check_confidence_and_score(topic_level, topic_scores)
            logger.info(f"confidence and score check result: {no_extra_explain}")
            
            # Detect whether a similar question had been asked before
            historic_answer, historic_answer_id = fetch_historic_data(standalone_question, no_extra_explain, vectorstore_name, tablespace_name)
            if historic_answer:
                if stream_handler:
                    logger.info(f"Streaming historic answer for grade {numeric_grade}")
                    for _, char in enumerate(historic_answer):
                        stream_handler(char)
                        time.sleep(0.02)
                    return ""
                else:
                    return historic_answer
            
            try:
                # Retrieve chunks for the current year
                logger.info("Retrieving chunks for current year...")
                current_year_chunks = get_chunks_for_current_year(detected_topic, numeric_grade, standalone_question)

                if current_year_chunks:
                    logger.debug(f"Found {len(current_year_chunks)} relevant chunks for {detected_topic}")
                else:
                    logger.warning(f"No relevant chunks found for {detected_topic} at level {numeric_grade}")                           
            except Exception as e:
                logger.error(f"Error getting chunks for {detected_topic}: {str(e)}")
                current_year_chunks = []
            
            # Get chunks from prior years
            try:
                lower_years_chunk = get_chunks_from_prior_years(detected_topic, numeric_grade, standalone_question)
                if lower_years_chunk:
                    logger.debug(f"Found {len(lower_years_chunk)} relevant chunks for {detected_topic}")
                else:
                    logger.warning(f"No relevant chunks found for {detected_topic} below level {numeric_grade}")      
            except Exception as e:
                logger.error(f"Error getting chunks from prior years for {detected_topic}: {str(e)}")
                lower_years_chunk = []

            # Process chunks (now they are text strings, not Document objects)
            processed_current_year_chunks = "\n\n".join(current_year_chunks) if current_year_chunks else ""
            processed_lower_years_chunks = "\n\n".join(lower_years_chunk) if lower_years_chunk else ""
            
            # Combine both chunks into one
            combined_chunks = processed_current_year_chunks
            if processed_current_year_chunks and processed_lower_years_chunks:
                combined_chunks += "\n\n"
            combined_chunks += processed_lower_years_chunks

            # Determine if student's question is covered in the retrieved chunks
            logger.info("Determining if question is covered in retrieved chunks...")
            chunk_coverage_prompt_text = load_prompt("chunk_coverage_prompt.txt")
            chunk_coverage_prompt = PromptTemplate.from_template(chunk_coverage_prompt_text)
            llm_decision = ChatOpenAI(model_name="gpt-4.1", temperature=0.0)
            coverage_answer_chain = chunk_coverage_prompt | llm_decision
            
            # Get the response from the LLM
            response_content = coverage_answer_chain.invoke({
                "student_question": standalone_question,
                "retrieved_chunks": combined_chunks
            }).content
            
            # Parse the response as a Python list
            try:
                coverage_answer = ast.literal_eval(response_content)
                logger.info(f"Coverage answer: {coverage_answer}")
                logger.info(f"Decision: {coverage_answer[0]}, Reasoning: {coverage_answer[1]}")
            except (ValueError, SyntaxError) as e:
                logger.error(f"Failed to parse coverage answer as list: {e}")
                logger.error(f"Raw response: {response_content}")
                # Default to Yes if parsing fails
                coverage_answer = ["Yes", "Failed to parse response, defaulting to covered."]
                
            if coverage_answer[0] == "No":
                response_message = f"This question is not covered in your textbook yet. I can only answer math questions related to your textbook content - happy to help with those! 😊"

                if stream_handler:
                    tokens = response_message.split()
                    for token in tokens:
                        stream_handler(token + " ")
                        time.sleep(0.05)
                    return ""
                else:
                    return response_message

            # Topic-based answer prompt
            topic_based_answer_prompt_text = load_prompt("topic_based_answer_prompt.txt")
            topic_prompt = PromptTemplate.from_template(topic_based_answer_prompt_text)

            # Check if we should use a single pass based on the previously calculated confidence check
            if no_extra_explain or numeric_grade == 9:
                try:
                    # Use streaming LLM to generate direct answer
                    answer_chain = topic_prompt | llm_final
                    logger.info("Generating single pass answer...")

                    final_answer = answer_chain.invoke({
                        "retrieved_chunks": processed_current_year_chunks,
                        "topic": detected_topic,
                        "student_question": standalone_question,
                    }).content.strip()
                    logger.info(f"Final Answer for topic-based question → {final_answer}")

                    if numeric_grade == 12 or numeric_grade == 11:
                        store_success = store_new_data(standalone_question, final_answer, no_extra_explain, vectorstore_name, tablespace_name,historic_answer_id)
                        if store_success:
                            logger.info(f"Successfully stored Q&A pair (single pass) for grade {numeric_grade}")
                        else:
                            logger.warning(f"Failed to store Q&A pair (single pass) for grade {numeric_grade}")     
                    return final_answer

                except Exception as e:
                    logger.error(f"Error generating answer: {str(e)}")
                    return "I'm sorry, I encountered an error while generating an answer. Please try again."
            
            # first pass
            try:
                logger.info("Generating first pass answer...")
                first_pass_chain = topic_prompt | llm_intermediate
                first_response = first_pass_chain.invoke({
                    "retrieved_chunks": processed_current_year_chunks,
                    "topic": detected_topic,
                    "student_question": standalone_question
                }).content.strip()
                logger.info(f"First Pass Answer for topic-based question → {first_response}")
            except Exception as e:
                logger.error(f"Error generating first pass answer: {str(e)}")
                return "I'm sorry, I encountered an error while generating an answer. Please try again."

            # Second pass
            logger.info("Generating second pass answer...")
            
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

            logger.info("Comparing answers...")
            compare_prompt_text = load_prompt("compare_prompt.txt")
            compare_prompt = PromptTemplate.from_template(compare_prompt_text)
            compare_chain = compare_prompt | llm_final
            
            # Use invoke method like single-pass to ensure consistent LaTeX rendering
            compare_answer = compare_chain.invoke({
                "first_pass_answer": first_response,
                "second_pass_answer": second_response,
                "student_question": standalone_question
            }).content.strip()
            
            logger.info(f"Compared answer: {compare_answer}")
            
            # Save generated answer and the student's question to both databases
            if numeric_grade == 12 or numeric_grade == 11:
                store_success = store_new_data(standalone_question, compare_answer, no_extra_explain, vectorstore_name, tablespace_name, historic_answer_id)
                if store_success:
                    logger.info(f"Successfully stored Q&A pair (two-pass) for grade {numeric_grade}")
                else:
                    logger.warning(f"Failed to store Q&A pair (two-pass) for grade {numeric_grade}")
            
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