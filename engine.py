from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
import pandas as pd
import os
from logger_config import setup_logger


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

# Test all log levels
logger.debug('Test DEBUG message')
logger.info('Test INFO message')
logger.warning('Test WARNING message')
logger.error('Test ERROR message')

def get_student_level(student_id):
    """Get student's grade from student_data.csv and convert grade to level.
    
    Args:
        student_id (str): The student's ID to look up
        
    Returns:
        str: The student's grade
    """
    try:
        df = pd.read_csv("student_data.csv")
        student_data = df[df["student_id"] == student_id]
        if student_data.empty:
            logger.warning(f"Student {student_id} not found in database")
            return None
        
        if student_data.iloc[0]["grade"] == 11:
            return "intermediate"
        elif student_data.iloc[0]["grade"] == 12:
            return "advanced"
        else:
            return "beginner"
    except Exception as e:
        logger.error(f"Error getting student grade: {str(e)}")
        return None

def get_student_topic_level(student_id, topic):
    """Get student's level (0 to 5) for a specific topic from student_data.csv.
    
    Args:
        student_id (str): The student's ID to look up
        topic (str): The topic to get the level for
        
    Returns:
        int: The student's level as an integer from 0 to 5
    """
    try:
        logger.info(f"Getting level for student {student_id} in topic {topic}")
        
        # Read student data - In future, this could be a database query
        df = pd.read_csv("student_data.csv")
        
        # Get the specific student's data
        student_data = df[df["student_id"] == student_id]
        
        if student_data.empty:
            logger.warning(f"Student {student_id} not found in database")
            return 0
        
        # Get the student's level for the topic
        student_level = student_data.iloc[0][topic]
        logger.info(f"Student {student_id} is at {student_level} level for {topic}")
        return student_level
        
    except Exception as e:
        logger.error(f"Error getting student level: {str(e)}")
        return 0

def get_relevant_chunks(topic, year, query):
    """Get relevant chunks for a topic up to and including the specified year level.
    
    Args:
        topic (str): The topic to search for
        year (str): The maximum year level ('beginner', 'intermediate', 'advanced')
        query (str): The search query
        
    Returns:
        list: Combined list of relevant chunks from all applicable levels
    """
    logger.info(f"Retrieving chunks for topic {topic} up to level {year}")
    
    # Map string levels to match build_index.py's file_level_mapping
    level_order = ['beginner', 'intermediate', 'advanced']
    if year not in level_order:
        logger.warning(f"Invalid level {year}, defaulting to beginner")
        year = 'beginner'
    max_level_idx = level_order.index(year)
    
    try:
        # Get chunks from all levels up to and including the student's level
        index_path = os.path.join('indexes', topic)
        if not os.path.exists(index_path):
            logger.error(f"No index found for topic {topic} at {index_path}")
            logger.debug(f"Available indexes: {os.listdir('indexes') if os.path.exists('indexes') else 'none'}")
            return []
            
        db = Chroma(persist_directory=index_path, embedding_function=OpenAIEmbeddings())
        
        try:
            # Check what metadata is actually in the database
            collection = db.get()
        except Exception as e:
            logger.warning(f"Could not inspect database metadata: {str(e)}")
        
        all_chunks = []
        
        # Get chunks for each level up to max_level
        for level in level_order[:max_level_idx + 1]:
            k_value = 2 if level != year else 3  # Get more results from current level
            
            try:
                # Use similarity search directly with metadata filter
                chunks = db.similarity_search(
                    query,
                    k=k_value,
                    filter={"level": level}  # This should match the metadata set in build_index.py
                )
                
                if chunks:
                    logger.debug(f"Found {len(chunks)} chunks for level {level}")
                    # Add score metadata if missing
                    for chunk in chunks:
                        if 'score' not in chunk.metadata:
                            chunk.metadata['score'] = 0.0
                    all_chunks.extend(chunks)
                else:
                    logger.debug(f"No chunks found for level {level}")
                    
            except Exception as e:
                logger.warning(f"Error retrieving chunks for level {level}: {str(e)}")
                continue
        
        logger.info(f"Retrieved total of {len(all_chunks)} chunks for topic {topic}")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks for topic {topic}: {str(e)}")
        return []

def get_topic(question):
    """Determine the math topics based on the question. 
    Returns:
        - None if it's a calculation question
        - 'overview' if it's asking about general capabilities
        - The most relevant topic otherwise
    """

    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        topic_prompt = PromptTemplate.from_template("""
            Analyze if the following question is:
            A) A regular math calculation/arithmetic (e.g., "what is 2+2", "calculate 15% of 80", "solve 3x + 5 = 11")
            B) A general inquiry about capabilities (e.g., "what can you teach me", "what topics do you cover", "what can I learn from you")
            C) A question about specific math topics/concepts that requires explanation

            If it's type A, respond with exactly: "calculation"
            If it's type B, respond with exactly: "overview"
            If it's type C, classify it into one of these topics that is the most relevant:

            - algebra: Linear equations, inequalities, binomial theorem
            - basics: Numbers, polynomials, coordinate geometry, real numbers, pair of linear equations, quadratic equations, complex numbers
            - geometry: Coordinate geometry, trigonometry, circles, surface areas, volumes, conic sections, three dimensional geometry
            - miscellaneous: Arithmetic progression, infinite series, sets, relations and functions, permutations and combinations, sequences and series
            - modelling: Mathematical modeling and applications
            - probability: Probability theory and applications
            - statistics: Data analysis, statistical methods, measures of central tendency

            Question: {question}
            
            For type C, respond with the **exact** topic name from the list above.
            e.g. "algebra" or "geometry"
            """)
        
        topic_chain = LLMChain(llm=llm, prompt=topic_prompt)
        response = topic_chain.run({"question": question}).strip().lower()
        
        if response == "calculation":
            return None
        elif response == "overview":
            return "overview"
        
        # Validate the topic
        valid_topics = ["algebra", "basics", "geometry", "miscellaneous", 
                       "modelling", "probability", "statistics"]
        if response in valid_topics:
            return response
        
        return "basics"
        
    except Exception as e:
        logger.error(f"Error detecting topics: {str(e)}")
        return "basics"  # Default to basics if detection fails

def rephrase_question(user_question: str, history: str) -> str:
    """Generate a standalone version of a follow-up question using chat history.

    Args:
        user_question (str): The raw question from the student.
        history (str): Concatenated chat history.

    Returns:
        str: Reformulated standalone question (falls back to original on error).
    """
    try:
        llm_rephrase = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        rephrase_prompt = PromptTemplate.from_template(
            """
            Given a chat history and the latest user question, reformulate the question as a standalone question.
            Pay special attention to:
            1. References to practice examples or exercises from previous responses
            2. Questions asking for answers, solutions or explanations of non-specified problems, which likely refer to the last example
            3. References to specific values or numbers mentioned in the history
            4. Questions that require information from parts of the chat history to answer

            If the question refers to a practice example or previous problem, include the full problem details in your reformulation.
            If the question requires information from parts of the chat history, include the relevant details in your reformulation.
            If the question is already standalone and doesn't reference the history, return the original question.

            Chat history:
            {history}

            Follow-up question:
            {user_question}

            Example reformulations:
            - Original: "what is the answer?" (after a practice example about a 45° triangle)
            Reformulated: "In the practice example with a right triangle with angle 45°, opposite side 4 units, and hypotenuse 5 units, what is the length of the adjacent side?"
            - Original: "I don't understand how to solve it"
            Reformulated: "How do I solve the problem about finding the adjacent side in a right triangle with angle 45°, opposite side 4 units, and hypotenuse 5 units?"
            - Original: "What did we talk about?"
            Reformulated: "What did we talk about? (previous topic was trigonometric functions)"
            """
        )
        llm_chain = LLMChain(llm=llm_rephrase, prompt=rephrase_prompt)
        return llm_chain.run({"user_question": user_question, "history": history}).strip()
    except Exception as e:
        logger.error(f"Error rephrasing question: {str(e)}")
        return user_question

def pipeline(student_id, user_question, history, stream_handler=None):
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
                calculation_prompt = PromptTemplate.from_template("""
                    You are an expert math tutor. Answer the following math calculation question.
                    Show your work step by step, but be concise. At the end, warn that AI is less reliable for calculations. Students should double-check the outputs.

                    Question:
                    {question}
                """)
                
                answer_chain = LLMChain(llm=llm_answer, prompt=calculation_prompt)
                response = answer_chain.run({"question": standalone_question}).strip()
                logger.info(f"Final Answer for calculation question → {response}")
                return response
            
            elif detected_topic == "overview":
                # It's a general inquiry about capabilities
                overview_prompt = PromptTemplate.from_template("""
                    Answer the student's question. Bellow is a list of topics you cover, use these as context when necessary:

                    - Algebra: Linear equations, inequalities, binomial theorem
                    - Basics: Numbers, polynomials, coordinate geometry, real numbers, pair of linear equations, quadratic equations, complex numbers
                    - Geometry: Coordinate geometry, trigonometry, circles, surface areas, volumes, conic sections, three dimensional geometry
                    - Miscellaneous: Arithmetic progression, infinite series, sets, relations and functions, permutations and combinations, sequences and series
                    - Mathematical Modeling: Applications of math to real-world problems
                    - Probability: Probability theory and applications
                    - Statistics: Data analysis, statistical methods, measures of central tendency

                    Keep the response friendly and encouraging, but concise.

                    Question:
                    {question}
                """)
                
                answer_chain = LLMChain(llm=llm_answer, prompt=overview_prompt)
                response = answer_chain.run({"question": standalone_question}).strip()
                logger.info(f"Final Answer for overview question → {response}")
                return response

            # Determine student's overall grade-based level (mapped from grade 11/12)
            grade_based_level = get_student_level(student_id) or "beginner"
            logger.debug(f"Student {student_id} is at grade {grade_based_level}")

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
            chunk_coverage_prompt = PromptTemplate.from_template("""
                You are a curriculum-aligned tutor assistant. A student has asked a question, and you have been given excerpts from their current textbooks. Your task is to decide whether the student's question can be reasonably answered based on the textbook content.

                Consider the following:
                - The question may not appear word-for-word in the textbook, but if the underlying concept or skill is explained, it counts as covered.
                - If the question is **on the same topic** but **requires knowledge that is not yet taught** in the retrieved material, consider it **not covered**.
                
                ---

                **Student Question:**
                {student_question}

                **Retrieved Textbook Content:**
                {retrieved_chunks}

                ---

                Answer **only** with one of the following:
                - `Yes` – if the question is clearly answerable using the retrieved content.
                - `No` – if the question is not covered or is too advanced based on this content.

                Keep in mind to return **exactly** with **only** the word `Yes` or `No` and nothing else.
            """)
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
            topic_prompt = PromptTemplate.from_template("""
                You are an intelligent math tutor helping students understand questions based on textbook materials. The student has asked a question that is covered by the curriculum and has a specific confidence level with the topic.

                Below is the student question, the relevant textbook excerpts, and the student's self-rated confidence on this topic (from 1 to 5):

                ---

                **Student Question:**
                {student_question}

                **Relevant Textbook Content:**
                {retrieved_chunks}

                **Topic:**
                {topic}

                **Student Confidence Level (1–5):**
                {confidence_level}

                ---

                Please answer the student's question **based on the textbook content and their confidence level**. 

                Adjust your explanation style based on the student's confidence level as follows. Make sure the depth, tone, vocabulary, and structure are meaningfully different at each level. Do not reuse the same explanation across levels.

                - **Confidence Level 1 (Beginner):**
                    - Assume the student has little to no background in the topic.
                    - Use very simple language and short sentences.
                    - Break down the explanation into small, clear steps.
                    - Define every math term (e.g., radius, slope).
                    - Use real-life analogies or visuals where possible.
                    - Do not use jargon or formal math notation unless you explain it carefully.

                - **Confidence Level 2 (Low):**
                    - Assume the student has seen the topic before but is still unsure.
                    - Use simple terms, but introduce basic math language with explanations.
                    - Provide step-by-step reasoning, with examples.
                    - Focus on building intuition and comfort with the process.

                - **Confidence Level 3 (Moderate):**
                    - Assume the student has a basic grasp and is looking to solidify understanding.
                    - Use standard math vocabulary and walk through logic clearly.
                    - Include some math notation or formulas with explanation.
                    - Give a clean, thorough explanation without over-explaining.

                - **Confidence Level 4 (High):**
                    - Assume the student understands the topic but may need clarity or refinement.
                    - Use formal math terms and concise language.
                    - Connect this concept to other related ideas or typical problems.
                    - Offer strategic tips or shortcuts when appropriate.

                - **Confidence Level 5 (Advanced):**
                    - Assume the student has a strong grasp and wants deeper insights.
                    - Be precise, efficient, and use technical language.
                    - Refer to related theorems, derivations, or alternate methods if relevant.
                    - Emphasize reasoning, abstraction, and generalization over step-by-step detail.

                Always adjust tone, detail, and vocabulary based on the level. Use examples or equations where helpful, but only at the appropriate level.

                **Do not repeat or mention the student’s question, confidence level, topic, or textbook content. Do not say "Since your confidence level is..."**

                Make sure the explanation is accurate and based only on the provided material. 
                Return only a clear, direct answer for the student's level, **not every level and DO NOT MENTION THE CONFIDENCE LEVEL**.
            """)

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