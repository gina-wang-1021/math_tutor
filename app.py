import streamlit as st
import numpy as np
from openai import OpenAI
from engine import generate_response
import pandas as pd
import csv

df = pd.read_csv("student_data.csv")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY2"])

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
            st.session_state.login_success = True
            st.session_state.login_failed = False
            st.session_state.student_id = student_id
            st.success("Login successful!")
            st.rerun()
        else:
            st.session_state.login_failed = True
            st.rerun()

# Main Chatting Page
else:
    student_name = df[df["student_id"] == st.session_state.student_id]["student_fname"].values[0]
    st.title(f"Hi {student_name}, let's learn math today!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input(f"Ask a question on math"):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            history_str = "\n".join([
                f"User: {m['content']}" if m["role"] == "user" else f"AI: {m['content']}"
                for m in st.session_state.messages
            ])

            stream = generate_response(st.session_state.user_grade, st.session_state.learning_topic, question, history_str)
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

