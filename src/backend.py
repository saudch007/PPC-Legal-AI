import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Optional
import streamlit as st
from pathlib import Path

# LangChain components for RAG
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # NEW: MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain # NEW: For simpler chain composition
from langchain.chains.combine_documents import create_stuff_documents_chain # NEW: For context stuffing
from langchain_core.runnables.history import RunnableWithMessageHistory # NEW: For conversation history
from langchain_community.chat_message_histories import ChatMessageHistory # NEW: To store messages
from langchain_core.runnables import chain as runnable_chain # For cleaner chain composition


load_dotenv()

# --- Configuration ---
PROJECT_ROOT = os.getcwd()  # Use current working directory for deployment compatibility
print("PROJECT_ROOT:", PROJECT_ROOT)
PDF_FILE_PATH = os.path.join(PROJECT_ROOT, "src", "data", "doc.pdf")
CHROMA_PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "src", "db")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    

def ingest_and_get_retriever() -> Optional[Chroma]:
   
    """
    Handles data ingestion (loading, chunking, embedding) and returns a ChromaDB instance.
    Checks if the vector store already exists to avoid re-ingestion.
    """
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set in backend.py.")
        return None

    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Error initializing OpenAI Embeddings: {e}")
        return None

    is_chroma_populated = False
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        if os.path.exists(os.path.join(CHROMA_PERSIST_DIRECTORY, "chroma.sqlite3")) or \
           os.path.exists(os.path.join(CHROMA_PERSIST_DIRECTORY, "collections")):
            is_chroma_populated = True

    if is_chroma_populated:
        print(f"ChromaDB already exists and is populated at '{CHROMA_PERSIST_DIRECTORY}'. Loading existing DB.")
        try:
            vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings_model)
            print(f"Loaded existing ChromaDB with {vector_store._collection.count()} documents.")
            return vector_store
        except Exception as e:
            print(f"Error loading existing ChromaDB from '{CHROMA_PERSIST_DIRECTORY}': {e}. Attempting re-ingestion.")
            pass

    print(f"\n--- Ingesting data as ChromaDB not found or empty or failed to load ---")
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: The PDF file '{PDF_FILE_PATH}' does not exist. Cannot ingest data.")
        return None

    loader = PyPDFLoader(PDF_FILE_PATH)
    pages: List[Document] = []
    try:
        for page in loader.lazy_load():
            if page.page_content.strip():
                pages.append(page)
            else:
                print(f"Warning: Page {page.metadata.get('page', 'N/A')} of '{os.path.basename(PDF_FILE_PATH)}' is empty or contains only whitespace. Skipping.")
    except Exception as e:
        print(f"Error loading PDF '{PDF_FILE_PATH}': {e}")
        return None

    if not pages:
        print(f"No pages were loaded from '{PDF_FILE_PATH}'. Cannot proceed with ingestion.")
        return None

    print(f"Successfully loaded {len(pages)} pages from the PDF for ingestion.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    document_chunks: List[Document] = text_splitter.split_documents(pages)
    print(f"Documents chunked into {len(document_chunks)} pieces.")

    print(f"Creating ChromaDB vector store in '{CHROMA_PERSIST_DIRECTORY}'...")
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    try:
        vector_store = Chroma.from_documents(
            documents=document_chunks,
            embedding=embeddings_model,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        print(f"ChromaDB vector store created and persisted with {vector_store._collection.count()} documents.")
        return vector_store
    except Exception as e:
        print(f"Error creating/persisting ChromaDB vector store during ingestion: {e}")
        return None


# - Modified get_rag_chain to support conversational history ---
def get_conversational_rag_chain():
    """
    Initializes and returns a conversational RAG chain with memory.
    """
    vector_store = ingest_and_get_retriever()

    if vector_store is None:
        print("Failed to get vector store. Cannot create conversational RAG chain.")
        return None

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Error initializing LLM: {e}. Check your OpenAI API key and internet connection.")
        return None

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

   
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if necessary and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}"), 
        ]
    )
   
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()


    # - Answer Generation Prompt (with history and context)
    
    qa_system_prompt = """You are a helpful and accurate legal assistant.
    Answer the user's question based ONLY on the following context and chat history.
    If the context and chat history do not contain the information to answer the question,
    state that you cannot find the answer in the provided legal document.
    Do not make up any information. Provide concise answers.

    Context: {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"), # Pass the chat history directly to the main QA prompt
            ("human", "{input}"), 
        ]
    )

    # 3. Document Stuffing Chain
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    
    retrieval_chain = RunnablePassthrough.assign(
        context=contextualize_q_chain | retriever 
    ).assign(
        answer=document_chain 
    )

    
    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        
        lambda session_id: ChatMessageHistory(session_id=session_id),
        input_messages_key="input", 
        history_messages_key="chat_history", 
        output_messages_key="answer", 
    )

    print("Conversational RAG chain constructed successfully.")
    return conversational_rag_chain


if __name__ == "__main__":
    print("Testing backend.py directly with conversational chain...")
    chain = get_conversational_rag_chain()
    if chain:
        session_id = "test_session_123"

        print("\n--- Turn 1 ---")
        query1 = "if i take a minor girl from one place to another. What does the Pakistan Penal Code say about this act?"
        print(f"Querying: '{query1}'")
        try:
            response1 = chain.invoke({"input": query1}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response1['answer']}")
        except Exception as e:
            print(f"Error during RAG chain invocation (Turn 1): {e}")

        print("\n--- Turn 2 ---")
        query2 = "What will be the punishment for this?" # Follow-up question
        print(f"Querying: '{query2}'")
        try:
            response2 = chain.invoke({"input": query2}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response2['answer']}")
        except Exception as e:
            print(f"Error during RAG chain invocation (Turn 2): {e}")

    else:
        print("Failed to get conversational RAG chain.")