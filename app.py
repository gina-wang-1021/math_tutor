import streamlit as st
import numpy as np
from openai import OpenAI
from engine import pipeline
import pandas as pd
from logger_config import setup_logger

# Initialize logger
logger = setup_logger('app')

df = pd.read_csv("student_data.csv")

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
        if student_id in df["student_id"].values:
            logger.info(f"Successful login for student {student_id}")
            st.session_state.login_success = True
            st.session_state.login_failed = False
            st.session_state.student_id = student_id
            st.success("Login successful!")
            st.rerun()
        else:
            logger.warning(f"Failed login attempt with ID {student_id}")
            st.session_state.login_failed = True
            st.rerun()

# Main Chatting Page
else:
    student_name = df[df["student_id"] == st.session_state.student_id]["student_fname"].values[0]
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

        # Handle assistant response
        with st.chat_message("assistant"):
            # Prepare chat history
            history_str = "\n".join([
                f"User: {m['content']}" if m["role"] == "user" else f"AI: {m['content']}"
                for m in st.session_state.messages
            ])
            logger.debug(f"Chat history length: {len(history_str.split())} words")
            
            try:
                logger.info("Calling pipeline")
                # Create placeholders for the hint and response
                hint_placeholder = st.empty()
                response_placeholder = st.empty()
                full_response = []

                # Show generating hint with subtle styling
                with hint_placeholder:
                    st.markdown("<div color='gray' style='font-size: 0.9em;'>⌛ Generating response...</div>", unsafe_allow_html=True)

                # Callback to handle streaming tokens
                def handle_token(token: str):
                    # Clear the hint on first token
                    if not full_response:
                        hint_placeholder.empty()
                    full_response.append(token)
                    # Join all tokens and display
                    response_placeholder.markdown(''.join(full_response), unsafe_allow_html=True)
                
                # Call pipeline with student_id and streaming handler
                response = pipeline(
                    student_id=st.session_state.student_id,
                    user_question=question,
                    history=history_str,
                    stream_handler=handle_token
                )
                
                if response:
                    # Get the complete response
                    complete_response = ''.join(full_response) if full_response else response
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": complete_response})
                    # Update the placeholder with the complete response
                    response_placeholder.markdown(complete_response, unsafe_allow_html=True)
                else:
                    # Clear the hint and show error
                    hint_placeholder.empty()
                    error_msg = "I couldn't generate a response. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                logger.error(f"Pipeline error: {str(e)}")
                # Clear the hint and show error
                hint_placeholder.empty()
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

