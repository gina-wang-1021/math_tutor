import streamlit as st
from openai import OpenAI
from engine import pipeline
import pandas as pd
from logger_config import setup_logger

# Initialize logger
logger = setup_logger('app')

df = pd.read_csv("official_db.csv")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "login_success" not in st.session_state:
    st.session_state.login_success = False
if "login_failed" not in st.session_state:
    st.session_state.login_failed = False
if "student_id" not in st.session_state:
    st.session_state.student_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Landing Page
if not st.session_state.login_success:
    st.title("Welcome to Your Study Buddy!")
    student_id = st.text_input("Please enter your student ID")

    if st.session_state.login_failed:
        st.error("Invalid student ID")
    
    if st.button("Start Learning"):
        student_id_str = str(student_id).strip()
        student_ids_in_df = [str(id).strip() for id in df["Student ID"].values]
        
        if student_id_str in student_ids_in_df:
            # Only log if this is the initial login, not during reruns
            if not st.session_state.login_success:
                logger.info(f"Successful login for student {student_id_str}")
            st.session_state.login_success = True
            st.session_state.login_failed = False
            st.session_state.student_id = student_id_str
            st.success("Login successful!")
            st.rerun()
        else:
            logger.warning(f"Failed login attempt with ID {student_id_str}")
            st.session_state.login_failed = True
            st.rerun()

# Main Chatting Page
else:
    matching_students = df[df["Student ID"].astype(str) == str(st.session_state.student_id)]
    
    if len(matching_students) > 0:
        first_name = matching_students["First name"].values[0]
        last_name = matching_students["Last name"].values[0]
        st.title(f"Hi {first_name} {last_name}, let's learn math today!")
    else:
        logger.error(f"No matching student found for ID: {st.session_state.student_id}")
        st.error("Error: Student record not found. Please log out and try again.")
        student_name = "Student"
        st.title(f"Hi {student_name}, let's learn math today!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if question := st.chat_input(f"Ask a question on math"):
        logger.info(f"New question from student {st.session_state.student_id}")
        logger.debug(f"Question: {question}")
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": question})

        # Handle assistant response - create a container for the response
        response_container = st.container()
        
        # Prepare chat history outside the container
        history_str = "\n".join([
            f"User: {m['content']}" if m["role"] == "user" else f"AI: {m['content']}"
            for m in st.session_state.messages
        ])
        logger.debug(f"Chat history length: {len(history_str.split())} words")
        
        # Create the assistant message container inside our custom container
        with response_container, st.chat_message("assistant"):
            # Create placeholders for the hint and response
            hint_placeholder = st.empty()
            response_placeholder = st.empty()
            full_response = []

            # Show generating hint
            with hint_placeholder:
                st.markdown("<div style='color: gray; font-size: 0.9em;'>⌛ Generating response...</div>", unsafe_allow_html=True)
            
            try:
                logger.info("Calling pipeline")

                # Callback to handle streaming tokens
                def handle_token(token: str):
                    if not full_response:
                        hint_placeholder.empty()
                    full_response.append(token)
                    response_placeholder.markdown(''.join(full_response), unsafe_allow_html=True)
                
                # Call pipeline with student_id and streaming handler
                response = pipeline(
                    student_id=st.session_state.student_id,
                    user_question=question,
                    history=history_str,
                    stream_handler=handle_token
                )
                
                # Check if we have streamed content or a direct response
                if full_response or response:
                    if not full_response:
                        hint_placeholder.empty()
                    
                    # Get the complete response - use streamed content if available, otherwise use direct response
                    complete_response = ''.join(full_response) if full_response else response
                    st.session_state.messages.append({"role": "assistant", "content": complete_response})
                    response_placeholder.markdown(complete_response, unsafe_allow_html=True)
                else:
                    # Only show error if both response is empty AND no content was streamed
                    hint_placeholder.empty()
                    error_msg = "I couldn't generate a response. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                logger.error(f"Pipeline error: {str(e)}")
                hint_placeholder.empty()
                st.error(f"An error occurred: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
