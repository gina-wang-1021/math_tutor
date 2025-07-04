import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from logger_config import setup_logger
from utilities.student_utils import get_student_level, get_confidence_level_and_score, check_confidence_and_score
from utilities.prompt_utils import load_prompt
from utilities.qa_utils import get_historic_answer, rephrase_question, store_qa_pair, MAX_L2_DISTANCE_THRESHOLD, HISTORIC_QA_K_NEIGHBORS
from utilities.retrieval_utils import get_chunks_for_current_year, get_chunks_from_prior_years, get_topic
import time

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
            logger.info("Rephrasing question...")
            standalone_question = rephrase_question(user_question, history)
            logger.info(f"Rephrased Question → {standalone_question}")
            
            # Detect topics from the rephrased question
            logger.info("Detecting topic...")
            detected_topic = get_topic(standalone_question)
            logger.info(f"Detected topic: {detected_topic if detected_topic else 'calculation only'}")

            # Initialize LLMs - one for intermediate steps (no streaming) and one for final answer (with streaming)
            llm_intermediate = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.5,
                streaming=False
            )
            
            llm_final = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.5,
                streaming=bool(stream_handler),
                callbacks=[StreamingCallbackHandler(stream_handler, delay=0.05)] if stream_handler else None
            )

            # Handle different types of questions
            if not detected_topic:
                # It's a calculation question
                logger.info("Processing calculation question...")
                calculation_prompt_text = load_prompt("calculation_answer_prompt.txt")
                calculation_prompt = PromptTemplate.from_template(calculation_prompt_text)
                
                # Use the streaming-enabled LLM for direct calculations (final answer)
                answer_chain = calculation_prompt | llm_final
                response = answer_chain.invoke({"question": standalone_question}).content.strip()
                logger.info(f"Final Answer for calculation question → {response}")
                return response
            
            elif detected_topic == "overview":
                # It's a general inquiry about capabilities
                logger.info("Processing overview question...")
                overview_prompt_text = load_prompt("overview_answer_prompt.txt")
                overview_prompt = PromptTemplate.from_template(overview_prompt_text)
                
                answer_chain = overview_prompt | llm_answer
                response = answer_chain.invoke({"question": standalone_question}).content.strip()
                logger.info(f"Final Answer for overview question → {response}")
                return response

            # Determine student's overall grade-based level (mapped from grade 11/12)
            logger.info("Determining student's year at school...")
            grade_based_level, numeric_grade = get_student_level(student_id)
            if grade_based_level is None and numeric_grade is None:
                grade_based_level = "ten"
                logger.info(f"Student {student_id}: Grade Based Level: {grade_based_level}, Numeric Grade: {numeric_grade}")

            # Detect whether a similar question had been asked before
            historic_answer = get_historic_answer(numeric_grade, standalone_question, historic_qa_l2_threshold)
            if historic_answer:
                # Stream the historic answer token by token if stream_handler is provided
                if stream_handler:
                    logger.info(f"Streaming historic answer for grade {numeric_grade}")
                    # Split the answer into tokens (words or smaller chunks)
                    for i, char in enumerate(historic_answer):
                        stream_handler(char)
                        # Add a small delay between tokens for a natural streaming effect
                        time.sleep(0.02)
                    return ""
                else:
                    return historic_answer

            # For topic-based questions, get the confidence level and scores for the topic
            logger.info("Getting student's confidence level and score for topic...")
            topic_level, topic_scores = get_confidence_level_and_score(student_id, detected_topic)
            logger.debug(f"Student confidence level and score for topic {detected_topic}: {topic_level}, {topic_scores}")
            
            try:
                # Retrieve chunks for the current year
                logger.info("Retrieving chunks for current year...")
                current_year_chunks = get_chunks_for_current_year(detected_topic, grade_based_level, standalone_question)

                if current_year_chunks:
                    logger.debug(f"Found {len(current_year_chunks)} relevant chunks for {detected_topic}")
                else:
                    logger.warning(f"No relevant chunks found for {detected_topic} at level {grade_based_level}")                           
            except Exception as e:
                logger.error(f"Error getting chunks for {detected_topic}: {str(e)}")
                current_year_chunks = []
            
            # Get chunks from prior years
            try:
                lower_years_chunk = get_chunks_from_prior_years(detected_topic, grade_based_level, standalone_question)
                if lower_years_chunk:
                    logger.debug(f"Found {len(lower_years_chunk)} relevant chunks for {detected_topic}")
                else:
                    logger.warning(f"No relevant chunks found for {detected_topic} below level {grade_based_level}")      
            except Exception as e:
                logger.error(f"Error getting chunks from prior years for {detected_topic}: {str(e)}")
                lower_years_chunk = []

            # Extract page_content from Document objects
            processed_current_year_chunks = "\n\n".join([chunk.page_content for chunk in current_year_chunks]) if current_year_chunks else ""
            processed_lower_years_chunks = "\n\n".join([chunk.page_content for chunk in lower_years_chunk]) if lower_years_chunk else ""
            
            # Combine both chunks into one
            combined_chunks = processed_current_year_chunks
            if processed_current_year_chunks and processed_lower_years_chunks:
                combined_chunks += "\n\n"
            combined_chunks += processed_lower_years_chunks

            # Determine if student's question is covered in the retrieved chunks
            logger.info("Determining if question is covered in retrieved chunks...")
            chunk_coverage_prompt_text = load_prompt("chunk_coverage_prompt.txt")
            chunk_coverage_prompt = PromptTemplate.from_template(chunk_coverage_prompt_text)
            llm_decision = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
            coverage_answer_chain = chunk_coverage_prompt | llm_decision
            
            # Get the response from the LLM
            response_content = coverage_answer_chain.invoke({
                "student_question": standalone_question,
                "retrieved_chunks": combined_chunks
            }).content
            
            # Parse the response as a Python list
            import ast
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
                response_message = f"This topic is not covered in your textbook yet. I'm happy to help with other questions you have! 😊"

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

            # Check confidence level and topic scores to see if second pass is needed
            if check_confidence_and_score(topic_level, topic_scores) or grade_based_level == "nine":
                try:
                    # Use streaming LLM to generate direct answer
                    answer_chain = topic_prompt | llm_final
                    logger.info("Generating single pass answer...")

                    final_answer = answer_chain.invoke({
                        "retrieved_chunks": processed_chunks,
                        "topic": detected_topic,
                        "student_question": standalone_question,
                    }).content.strip()
                    logger.info(f"Final Answer for topic-based question → {final_answer}")

                    if numeric_grade in [11, 12]:
                        store_success = store_qa_pair(numeric_grade, standalone_question, final_answer)
                        if store_success:
                            logger.info(f"Successfully stored Q&A pair for grade {numeric_grade} (single pass)")
                        else:
                            logger.warning(f"Failed to store Q&A pair for grade {numeric_grade}")     
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
            
            processed_chunks = "\n\n".join([chunk.page_content for chunk in lower_years_chunk]) if lower_years_chunk else ""
            
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
            compare_answer = compare_chain.invoke({
                "first_pass_answer": first_response,
                "second_pass_answer": second_response,
                "student_question": standalone_question
            }).content.strip()
            logger.info(f"Compared answer: {compare_answer}")
            
            # Save generated answer and the student's question to both databases
            if numeric_grade in [11, 12]:
                store_success = store_qa_pair(numeric_grade, standalone_question, compare_answer)
                if store_success:
                    logger.info(f"Successfully stored Q&A pair for grade {numeric_grade} (two-pass)")
                else:
                    logger.warning(f"Failed to store Q&A pair for grade {numeric_grade}")
            
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