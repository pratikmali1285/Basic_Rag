"""Streamlit chat UI for the HR Policy Assistant.

Run with:  streamlit run app.py
"""

import streamlit as st

from hr_assistant.pipeline import ask, build_hr_assistant

st.set_page_config(page_title="HR Policy Assistant", page_icon="💼")
st.title("💼 HR Policy Assistant")
st.caption("Ask questions about the company's HR policy.")


@st.cache_resource(show_spinner="Building the HR assistant...")
def get_agent():
    """Build the agent once and reuse it across reruns and sessions."""
    return build_hr_assistant()


try:
    agent = get_agent()
except ValueError as error:  # e.g. a missing API key in .env
    st.error(str(error))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("About")
    st.write("Answers come from the HR policy document, checked by input/output guardrails.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask an HR policy question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask(agent, question)
            except Exception as error:
                answer = f"Sorry, something went wrong: {error}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
