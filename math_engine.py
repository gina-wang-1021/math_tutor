from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from logger_config import setup_logger
from utilities.student_utils import get_confidence_level_and_score, check_confidence_and_score
from utilities.prompt_utils import load_prompt
from utilities.qa_utils import rephrase_question
from utilities.postgre_utils import insert_answer
from utilities.retrieval_utils_pinecone import get_chunks_for_current_year, get_chunks_from_prior_years, get_topic
import time
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
        student_id = student_data["Username"]
        if student_data["Customization"] == "1":
            customization = True
        else:
            customization = False
        
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
            topic_level, topic_scores = get_confidence_level_and_score("Math Score", student_data, detected_topic)
            logger.debug(f"Student confidence level and score for topic {detected_topic}: {topic_level}, {topic_scores}")
            
            # Check confidence level and topic scores
            no_extra_explain = check_confidence_and_score(topic_level, topic_scores)
            logger.info(f"confidence and score check result: {no_extra_explain}")
            
            current_year_chunks = []
            lower_years_chunk = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all tasks in parallel
                current_year_future = executor.submit(get_chunks_for_current_year, detected_topic, standalone_question)
                prior_years_future = executor.submit(get_chunks_from_prior_years, detected_topic, standalone_question)
                
                # Load prompt while other tasks are running
                topic_based_answer_prompt_text = load_prompt("topic_based_answer_prompt.txt")
                topic_prompt = PromptTemplate.from_template(topic_based_answer_prompt_text)
                
                # Get the knowledge chunks
                logger.info("Retrieving knowledge chunks...")
                
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

            # Process chunks
            processed_current_year_chunks = "\n\n".join(current_year_chunks) if current_year_chunks else ""
            processed_lower_years_chunks = "\n\n".join(lower_years_chunk) if lower_years_chunk else ""

            if not customization:
                try:
                    answer_chain = topic_prompt | llm_final
                    logger.info("Generating non-customized answer...")

                    non_customized_response = answer_chain.invoke({
                        "retrieved_chunks": processed_current_year_chunks,
                        "topic": detected_topic,
                        "student_question": standalone_question,
                    }).content.strip()
                    logger.info(f"Non-customized Answer → {non_customized_response}")
                    response_id = insert_answer(no_extra_explain, non_customized_response)
                    logger.info(f"Stored answer for id {response_id}")
                    return non_customized_response
                except Exception as e:
                    logger.error(f"Error generating non-customized answer: {str(e)}")
                    return "I'm sorry, I encountered an error while generating an answer. Please try again."
            
            # First pass
            try:
                answer_chain = topic_prompt | llm_intermediate
                logger.info("Generating first pass answer...")

                first_response = answer_chain.invoke({
                    "retrieved_chunks": processed_current_year_chunks,
                    "topic": detected_topic,
                    "student_question": standalone_question,
                }).content.strip()
                logger.info(f"First Pass Answer → {first_response}")
            except Exception as e:
                logger.error(f"Error generating answer: {str(e)}")
                return "I'm sorry, I encountered an error while generating an answer. Please try again."

            # Second pass
            if no_extra_explain:
                logger.info("Generating simplified explanation")
                simplified_prompt_text = load_prompt("simplified_prompt.txt")
                simplified_prompt = PromptTemplate.from_template(simplified_prompt_text)
                answer_chain = simplified_prompt | llm_final

                simplified_response = answer_chain.invoke({
                    "student_question": standalone_question,
                    "first_pass_answer": first_response
                }).content.strip()
                logger.info(f"Simplified Answer → {simplified_response}")
                try:
                    response_id = insert_answer(no_extra_explain, simplified_response)
                    logger.info(f"Stored answer for id {response_id}")
                except Exception as e:
                    logger.error(f"Error inserting answer: {str(e)}")
                    return "I'm sorry, I encountered an error while generating an answer. Please try again."
                return simplified_response
            else:
                try:
                    logger.info("Generating enhanced explanation")
                    explain_prompt_text = load_prompt("explain_prompt.txt")
                    explain_prompt = PromptTemplate.from_template(explain_prompt_text)
                    answer_chain = explain_prompt | llm_intermediate

                    enhanced_response = answer_chain.invoke({
                        "retrieved_chunks": processed_lower_years_chunks,
                        "first_pass_answer": first_response,
                        "student_question": standalone_question
                    }).content.strip()
                    logger.info(f"Enhanced Answer → {enhanced_response}")
                except Exception as e:
                    logger.error(f"Error generating answer: {str(e)}")
                    return "I'm sorry, I encountered an error while generating an answer. Please try again."

                logger.info("Comparing answers...")
                compare_prompt_text = load_prompt("compare_prompt.txt")
                compare_prompt = PromptTemplate.from_template(compare_prompt_text)
                compare_chain = compare_prompt | llm_final
                
                compare_answer = compare_chain.invoke({
                    "first_pass_answer": first_response,
                    "second_pass_answer": enhanced_response,
                    "student_question": standalone_question
                }).content.strip()
                logger.info(f"Compared answer: {compare_answer}")
                try:
                    response_id = insert_answer(no_extra_explain, compare_answer)
                    logger.info(f"Stored answer for id {response_id}")
                except Exception as e:
                    logger.error(f"Error inserting answer: {str(e)}")
                return compare_answer

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return "I'm having trouble accessing the learning materials. Please try again later."
            
    except Exception as e:
        logger.error(f"Error processing your question: {str(e)}")
        return "I'm having trouble understanding your question. Could you please rephrase it?"