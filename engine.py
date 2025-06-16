from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain 
import pandas as pd
from logger_config import setup_logger

# Initialize logger
logger = setup_logger('engine')

def get_student_level(student_id, topic):
    """Get student's level (0,1,2) for a specific topic from student_data.csv.
    
    Args:
        student_id (str): The student's ID to look up
        topic (str): The topic to get the level for
        
    Returns:
        str: The student's level as 'beginner', 'intermediate', or 'advanced'
    """
    try:
        logger.info(f"Getting level for student {student_id} in topic {topic}")
        
        # Read student data - In future, this could be a database query
        df = pd.read_csv("student_data.csv")
        
        # Get the specific student's data
        student_data = df[df["student_id"] == student_id]
        
        if student_data.empty:
            logger.warning(f"Student {student_id} not found in database")
            return "beginner"
        
        # Get the student's level for the topic
        student_level = student_data.iloc[0][topic]
        
        # Convert numeric level to string format
        level_map = {
            0: "beginner",
            1: "intermediate",
            2: "advanced"
        }
        
        level = level_map.get(student_level, "beginner")
        logger.info(f"Student {student_id} is at {level} level for {topic}")
        return level
        
    except Exception as e:
        logger.error(f"Error getting student level: {str(e)}")
        return "beginner"

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
    
    # Map year strings to numeric levels
    level_map = {
        'beginner': 0,
        'intermediate': 1,
        'advanced': 2
    }
    max_level = level_map.get(year, 0)
    logger.debug(f"Max numeric level: {max_level}")
    
    try:
        # Get chunks from all levels up to and including the student's level
        db = Chroma(persist_directory=f"indexes/{topic}", embedding_function=OpenAIEmbeddings())
        all_chunks = []
        
        # Adjust k based on level to get more results from higher levels
        for level in range(max_level + 1):
            k_value = 2 if level < max_level else 3  # Get more results from the current level
            logger.debug(f"Retrieving {k_value} chunks for level {level}")
            
            retriever = db.as_retriever(search_kwargs={"k": k_value, "filter": {"level": level}})
            chunks = retriever.invoke(query)
            
            if chunks:
                logger.debug(f"Found {len(chunks)} chunks for level {level}")
                all_chunks.extend(chunks)
            else:
                logger.debug(f"No chunks found for level {level}")
        
        logger.info(f"Retrieved total of {len(all_chunks)} chunks for topic {topic}")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks for topic {topic}: {str(e)}")
        return []

def get_topic(question):
    """Determine the math topics based on the question. 
    Returns:
        - Empty list [] if it's a calculation question
        - ['overview'] if it's asking about general capabilities
        - List of relevant topics otherwise
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
        If it's type C, classify it into one or more of these topics:

        - algebra: Linear equations, inequalities, binomial theorem
        - basics: Numbers, polynomials, coordinate geometry, real numbers, pair of linear equations, quadratic equations, complex numbers
        - geometry: Coordinate geometry, trigonometry, circles, surface areas, volumes, conic sections, three dimensional geometry
        - miscellaneous: Arithmetic progression, infinite series, sets, relations and functions, permutations and combinations, sequences and series
        - modelling: Mathematical modeling and applications
        - probability: Probability theory and applications
        - statistics: Data analysis, statistical methods, measures of central tendency

        Question: {question}
        
        For type C, reply with a comma-separated list of relevant topic names in lowercase, ordered by relevance (most relevant first).
        Include ALL relevant topics, but limit to max 3 most relevant ones.
        Example response format for type C: "geometry,algebra,basics"
        """)
        
        topic_chain = LLMChain(llm=llm, prompt=topic_prompt)
        response = topic_chain.run({"question": question}).strip().lower()
        
        if response == "calculation":
            return []
        elif response == "overview":
            return ["overview"]
            
        detected_topics = response.split(",")
        
        # Validate the topics
        valid_topics = ["algebra", "basics", "geometry", "miscellaneous", 
                       "modelling", "probability", "statistics"]
        validated_topics = [topic.strip() for topic in detected_topics if topic.strip() in valid_topics]
        
        # Return validated topics or default to basics if none are valid
        return validated_topics if validated_topics else ["basics"]
        
    except Exception as e:
        logger.error(f"Error detecting topics: {str(e)}")
        return ["basics"]  # Default to basics if detection fails

def pipeline(student_id, user_question, history):
    try:
        logger.info(f"Processing question for student {student_id}")
        logger.debug(f"Original question: {user_question}")
        logger.debug(f"History length: {len(history.split()) if history else 0} words")
        
        # Initialize LLM for rephrasing
        llm_rephrase = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        rephrase_prompt = PromptTemplate.from_template("""
        Given a chat history and the latest user question, reformulate the question as a standalone question.
        Pay special attention to:
        1. References to practice examples or exercises from previous responses
        2. Questions asking for answers, solutions or explanations of non-specified problems, which likely refer to the last example
        3. References to specific values or numbers mentioned in the history
        4. Questions that require information from parts of the chat history to answer

        If the question refers to a practice example or previous problem, include the full problem details in your reformulation.
        If the question requires information from parts of the chat history, include the relevant details in your reformulation.
        If the question is already standalone and doesn't reference the history, return it as-is.

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
        """)

        # Rephrase the question with context
        try:
            logger.info("Rephrasing question with context")
            llm_chain = LLMChain(llm=llm_rephrase, prompt=rephrase_prompt)
            standalone_question = llm_chain.run({
                "user_question": user_question,
                "history": history
            })
            logger.info(f"Rephrased Question → {standalone_question}")
            
            # Detect topics from the rephrased question
            detected_topics = get_topic(standalone_question)
            logger.info(f"Detected topics: {', '.join(detected_topics) if detected_topics else 'calculation only'}")

            # Initialize LLM for answering
            llm_answer = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

            # Handle different types of questions
            if not detected_topics:
                # It's a calculation question
                calculation_prompt = PromptTemplate.from_template("""
                You are an expert math tutor. Answer the following math calculation question.
                Show your work step by step, but be concise. At the end, warn that AI is less reliable for calculations and students should double-check the outputs.

                Question:
                {question}
                """)
                
                answer_chain = LLMChain(llm=llm_answer, prompt=calculation_prompt)
                response = answer_chain.run({"question": standalone_question})
                logger.info(f"Final Answer for calculation question → {response}")
                return response
            
            elif detected_topics == ["overview"]:
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
                response = answer_chain.run({"question": standalone_question})
                logger.info(f"Final Answer for overview question → {response}")
                return response

            # For topic-based questions, retrieve relevant documents
            all_docs = []
            for topic in detected_topics:
                # Get student level for this topic
                year = get_student_level(student_id, topic)
                logger.info(f"Student {student_id} level for {topic}: {year}")
                
                # Get relevant documents for each topic
                try:
                    topic_docs = get_relevant_chunks(topic, year, standalone_question)
                    all_docs.extend(topic_docs)
                except Exception as e:
                    logger.error(f"Error retrieving documents for topic {topic}: {str(e)}")

            # Sort documents by relevance score and level
            # Prioritize higher relevance scores, but give a small boost to higher level content
            all_docs = sorted(all_docs, 
                            key=lambda x: (getattr(x, 'score', 0) + 
                                         x.metadata.get('level', 0) * 0.1), 
                            reverse=True)
            
            # Get top 5 chunks and format them with level indicators
            selected_docs = all_docs[:5]
            level_names = {0: 'Beginner', 1: 'Intermediate', 2: 'Advanced'}
            
            chunks = []
            for doc in selected_docs:
                level = doc.metadata.get('level', 0)
                level_name = level_names.get(level, 'Beginner')
                chunks.append(f"[{level_name}] {doc.page_content}")
                
            chunks = "\n\n".join(chunks)

            # Get student levels for all detected topics
            topic_levels = {topic: get_student_level(student_id, topic) for topic in detected_topics}
            student_level_info = "\n".join([f"- {topic}: {level}" for topic, level in topic_levels.items()])
            
            # Topic-based answer prompt
            topic_prompt = PromptTemplate.from_template("""
            You are an expert math tutor with years of experience in helping students understand mathematical concepts.
            Adapt your explanation based on the student's current understanding level of each topic, but do not explicitly mention their level.

            Student's Current Understanding Levels (for your reference only, do not mention in response):
            {student_levels}

            Internal guidelines (do not mention these in response):
            - Beginner: Use simple language, basic examples, avoid jargon unless specifically prompted for
            - Intermediate: Introduce technical terms with explanations
            - Advanced: Use proper terminology, deeper theory

            Follow these steps (but do not list them in your response):
            1. Give a clear, direct explanation at appropriate complexity
            2. Break down complex ideas as needed
            3. Provide relevant examples
            4. Address common misconceptions

            If the answer isn't fully covered in the context:
            - Note what information is missing
            - Explain what you can with available information
            - Suggest related topics to review

            Context (with difficulty levels marked - do not mention these markers in your response):
            {chunks}

            Student's Question:
            {question}

            Important: Answer directly without mentioning the student's level or using phrases like "as a beginner" or "for your level".
            """)

            # Generate answer
            try:
                answer_chain = LLMChain(llm=llm_answer, prompt=topic_prompt)
                logger.info("Generating answer for topic-based question")

                response = answer_chain.run({
                    "chunks": chunks,
                    "question": standalone_question,
                    "student_levels": student_level_info
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