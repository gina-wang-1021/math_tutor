import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from scripts.logger_config import setup_logger
from utilities.student_utils import get_student_level, get_confidence_level_and_score, check_confidence_and_score
from utilities.prompt_utils import load_prompt
from utilities.qa_utils import get_historic_answer, rephrase_question, MAX_L2_DISTANCE_THRESHOLD, HISTORIC_QA_K_NEIGHBORS
from utilities.retrieval_utils import get_chunks_for_current_year, get_chunks_from_prior_years, get_topic
import time

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
                logger.info("Processing calculation question...")
                calculation_prompt_text = load_prompt("calculation_answer_prompt.txt")
                calculation_prompt = PromptTemplate.from_template(calculation_prompt_text)
                
                answer_chain = calculation_prompt | llm_answer
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
            if grade_based_level is None and numeric_grade is None: # student_id not found or other error in get_student_level
                grade_based_level = "beginner"
                logger.info(f"Student {student_id}: Grade Based Level: {grade_based_level}, Numeric Grade: {numeric_grade}")

            # Detect whether a similar question had been asked before
            # historic_answer = get_historic_answer(numeric_grade, standalone_question, historic_qa_l2_threshold)
            # if historic_answer:
            #     return historic_answer

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
                
            # Extract page_content from Document objects
            processed_chunks = "\n\n".join([chunk.page_content for chunk in current_year_chunks]) if current_year_chunks else ""

            # Determine if student's question is covered in the retrieved chunks
            logger.info("Determining if question is covered in retrieved chunks...")
            chunk_coverage_prompt_text = load_prompt("chunk_coverage_prompt.txt")
            chunk_coverage_prompt = PromptTemplate.from_template(chunk_coverage_prompt_text)
            llm_decision = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
            coverage_answer_chain = chunk_coverage_prompt | llm_decision
            coverage_answer = coverage_answer_chain.invoke({
                "student_question": standalone_question,
                "retrieved_chunks": processed_chunks
            }).content.strip()
            logger.info(f"Coverage answer: {coverage_answer}")

            if coverage_answer.strip() == "No" or coverage_answer.strip() == "no":
                # Create the response message
                response_message = f"This topic is not covered in your textbook yet. I'm happy to help with other questions you have! 😊"
                
                # If stream handler is provided, simulate token-by-token streaming
                if stream_handler:
                    tokens = response_message.split()
                    for token in tokens:
                        stream_handler(token + " ")
                        time.sleep(0.05)  # Small delay between tokens
                    return ""
                else:
                    # If no stream handler, return the full message as before
                    return response_message
            
            # Topic-based answer prompt
            topic_based_answer_prompt_text = load_prompt("topic_based_answer_prompt.txt")
            topic_prompt = PromptTemplate.from_template(topic_based_answer_prompt_text)

            # Generate answer
            try:
                answer_chain = topic_prompt | llm_answer
                logger.info("Generating answer for topic-based question")

                first_response = answer_chain.invoke({
                    "retrieved_chunks": processed_chunks,
                    "topic": detected_topic,
                    "student_question": standalone_question,
                    "confidence_level": topic_level,
                    "topic_scores": topic_scores
                }).content.strip()
                logger.info(f"First Pass Answer for topic-based question → {first_response}")
                
            except Exception as e:
                logger.error(f"Error generating answer: {str(e)}")
                return "I'm sorry, I encountered an error while generating an answer. Please try again."

            # Check confidence level and topic scores to see if second pass is needed
            if check_confidence_and_score(topic_level, topic_scores) or grade_based_level == "beginner":

                # TODO: Save generated answer and the student's question to both databases
                
                return first_response
            
            # Second pass
            try:
                logger.info("Generating second pass answer...")
                lower_years_chunk = get_chunks_from_prior_years(detected_topic, grade_based_level, standalone_question)
            
                if lower_years_chunk:
                    logger.debug(f"Found {len(lower_years_chunk)} relevant chunks for {detected_topic}")
                else:
                    logger.warning(f"No relevant chunks found for {detected_topic} at level {grade_based_level}")  
            except Exception as e:
                logger.error(f"Error getting chunks from prior years: {str(e)}")

                # TODO: Save generated answer and the student's question to both databases

                return first_response
            
            # Extract page_content from Document objects
            processed_chunks = "\n\n".join([chunk.page_content for chunk in lower_years_chunk]) if lower_years_chunk else ""
            
            logger.info("Generating detail explanation for topic-based question...")
            explain_prompt_text = load_prompt("explain_prompt.txt")
            explain_prompt = PromptTemplate.from_template(explain_prompt_text)
            answer_chain = explain_prompt | llm_answer

            second_response = answer_chain.invoke({
                "retrieved_chunks": processed_chunks,
                "first_pass_answer": first_response,
                "student_question": standalone_question
            }).content.strip()
            logger.info(f"Second Pass Answer for topic-based question → {second_response}")

            # TODO: Save generated answer and the student's question to both databases
            
            logger.info("Comparing answers...")
            compare_prompt_text = load_prompt("compare_prompt.txt")
            compare_prompt = PromptTemplate.from_template(compare_prompt_text)
            compare_chain = compare_prompt | llm_decision
            compare_answer = compare_chain.invoke({
                "first_pass_answer": first_response,
                "second_pass_answer": second_response,
                "student_question": standalone_question
            }).content.strip()
            logger.info(f"Compared answer: {compare_answer}")
            
            # TODO: Save generated answer and the student's question to both databases
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

if __name__ == "__main__":
    pipeline("beginner", "basics", "what is 10+10", "")