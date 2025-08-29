import streamlit as st
from openai import OpenAI
from math_engine import pipeline as math_pipeline
from econ_engine import pipeline as econ_pipeline
from logger_config import setup_logger

# Initialize logger
logger = setup_logger('app')

student_db = st.secrets["rows"]

# Create a lookup dictionary for faster authentication (supports multiple entries per username)
student_lookup = {}
for student in student_db:
    username_key = str(student["Username"]).lower()
    if username_key not in student_lookup:
        student_lookup[username_key] = []
    student_lookup[username_key].append(student)
valid_student_ids = set(student_lookup.keys())

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "login_success" not in st.session_state:
    st.session_state.login_success = False
if "login_failed" not in st.session_state:
    st.session_state.login_failed = False
if "student_data" not in st.session_state:
    st.session_state.student_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Landing Page
if not st.session_state.login_success:
    st.title("Welcome to Your Study Buddy!")
    student_id = st.text_input("Please enter your username")
    password = st.text_input("Please enter your password", type="password")

    if st.session_state.login_failed:
        st.error("Invalid username or password")
    
    if st.button("Start Learning"):
        student_id_str = str(student_id).strip()
        password_str = str(password).strip()
        
        # Check if student ID exists and find matching password
        matching_student = None
        if student_id_str in valid_student_ids:
            # Check up to 2 entries for this username
            student_entries = student_lookup[student_id_str][:2]  # Limit to 2 entries
            for student_entry in student_entries:
                if student_entry.get("Password") == password_str:
                    matching_student = student_entry
                    break
        
        if matching_student:
            # Only log if this is the initial login, not during reruns
            if not st.session_state.login_success:
                logger.info(f"Successful login for student {student_id_str}")
            st.session_state.login_success = True
            st.session_state.login_failed = False
            st.session_state.student_data = matching_student
            st.success("Login successful!")
            st.rerun()
        else:
            logger.warning(f"Failed login attempt with username {student_id_str}")
            st.session_state.login_failed = True
            st.rerun()

# Main Chatting Page
else:
    # Get current student data
    current_student = st.session_state.student_data
    student_name = f"{current_student['First name'].lower().capitalize()} {current_student['Last name'].lower().capitalize()}"
    learning_topic = current_student.get('Learning Topic', 'Math')  # Default to math if not specified
    # Display personalized greeting based on learning topic
    if learning_topic.lower() == 'math':
        st.title(f"Hi {current_student['First name'].lower().capitalize()}, let's learn math today!")
        subject_name = "Math"
        pipeline_func = math_pipeline
    elif learning_topic.lower() == 'econ':
        st.title(f"Hi {current_student['First name'].lower().capitalize()}, let's learn economics today!")
        subject_name = "Economics"
        pipeline_func = econ_pipeline
    else:
        st.title(f"Hi {current_student['First name'].lower().capitalize()}, let's start learning!")
        subject_name = "your subject"
        pipeline_func = math_pipeline
    
    # Display student info in sidebar
    with st.sidebar:
        st.write(f"**Username:** {st.session_state.student_data['Username']}")
        st.write(f"**Name:** {student_name}")
        st.write(f"**Learning Topic:** {subject_name}")
        if st.button("Logout"):
            logger.info(f"Logging out for student {st.session_state.student_data['Username']}")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if question := st.chat_input(f"Ask a question on {subject_name}"):
        logger.info(f"New question from student {st.session_state.student_data['Username']}")
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
                
                # Call appropriate pipeline based on learning topic
                response = pipeline_func(
                    student_data=st.session_state.student_data,
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
