import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from llm import get_ai_message
import os

st.title("Streamlit 기본예제")

st.write("소득세와 관련된 모든것을 답변해 드립니다.")

if "message_list" not in st.session_state: # 누적된 데이터값을 보관(기억)해둠
    st.session_state.message_list = []
    print(f"before == {st.session_state.message_list}")

for message in st.session_state.message_list: 
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input(placeholder='소득세에 관련된 궁금한 내용들을 말씀하세요.'):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다."):
        ai_response = get_ai_message(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
    st.session_state.message_list.append({"role": "ai", "content": ai_message})