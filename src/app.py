import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.messages import AIMessage, HumanMessage 


from backend import get_conversational_rag_chain 

# Load environment variables 
load_dotenv()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Legal AI Assistant (Pakistan Penal Code)", layout="centered")

st.title("⚖️ Legal AI Assistant: Pakistan Penal Code")
st.markdown("Ask questions about the Pakistan Penal Code, and I'll try to find answers from the document. This assistant now has **memory** to understand follow-up questions!")


@st.cache_resource
def load_rag_chain():
    """Load the conversational RAG chain and display loading status."""
    with st.spinner("Initializing legal AI assistant... This may take a moment (loading data and LLM)."):
        
        chain = get_conversational_rag_chain()
        if chain:
            st.success("Legal AI Assistant is ready!")
        else:
            st.error("Failed to initialize Legal AI Assistant. Check backend logs and API key.")
    return chain

# Load the RAG chain with history support
rag_chain_with_history = load_rag_chain()

# Initialize chat history in Streamlit's session state
# This is crucial for maintaining conversation turns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_user_session" 


for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


if rag_chain_with_history:
   
    user_query = st.chat_input(
        "Ask a question about the Pakistan Penal Code...",
        key="user_query_input" 
    )

    if user_query:
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

       
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    
                    response = rag_chain_with_history.invoke(
                        {"input": user_query, "chat_history": st.session_state.chat_history},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )

                    # Extract the actual answer content from the response dictionary
                    ai_response_content = response["answer"] if isinstance(response, dict) and "answer" in response else str(response)

                    st.markdown(ai_response_content)
                    # Add AI's response to chat history
                    st.session_state.chat_history.append(AIMessage(content=ai_response_content))

                    # Optional: Display a note about sources (can be expanded if backend provides sources)
                    st.markdown("---")
                    st.info("Answer based on the provided Pakistan Penal Code document.")

                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
                    st.info("Please check your OpenAI API key, internet connection, or try a different query.")
        # Ensure the chat input is cleared after submission (st.chat_input handles this automatically)
else:
    # Message displayed if rag_chain_with_history is None (initialization failed)
    st.warning("Legal AI Assistant could not be loaded. Please check the console for backend errors and ensure your API key is valid.")

st.markdown("---")
st.caption("Developed by Saud")