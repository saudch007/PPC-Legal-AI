import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Optional
import streamlit as st
from pathlib import Path

# LangChain components for RAG
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# FIX: Corrected import as per deprecation warning
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# Explicitly import for clarity
from langchain.chains import create_history_aware_retriever


# Load environment variables (important for local development)
load_dotenv()

# --- Configuration for Paths ---
# Get the directory where backend.py is located (e.g., /mount/src/PPC-Legal-AI/src)
current_file_dir = Path(__file__).parent
# Go up one level from src/ to the project root (e.g., /mount/src/PPC-Legal-AI/)
PROJECT_ROOT_DIR = current_file_dir.parent

# Define paths relative to the calculated PROJECT_ROOT_DIR
# This correctly points to 'data' and 'db' at the repository root level
PDF_FILE_PATH = PROJECT_ROOT_DIR / "data" / "Pakistan_Penal_Code.pdf"
CHROMA_PERSIST_DIRECTORY = PROJECT_ROOT_DIR / "db"
# Define directory for potentially scraped data (if you decide to use it)
SCRAPED_DATA_DIR = PROJECT_ROOT_DIR / "scraped_legal_data"

# Convert Path objects to strings for functions that require them
pdf_path_str = str(PDF_FILE_PATH)
chroma_dir_str = str(CHROMA_PERSIST_DIRECTORY)
scraped_data_dir_str = str(SCRAPED_DATA_DIR)


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- Corrected OpenAI API Key Handling (Global Initialization) ---
OPENAI_API_KEY: Optional[str] = None

# Try loading from local .env first (for local development)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print("DEBUG: Loaded OPENAI_API_KEY from local .env file.")
else:
    # If not found in .env, try Streamlit secrets (for deployment)
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        print("DEBUG: Loaded OPENAI_API_KEY from Streamlit secrets.")
    else:
        # If still not found, raise an error to stop execution early
        raise ValueError("CRITICAL ERROR: OPENAI_API_KEY is not set. Please set it in your local .env file or Streamlit Cloud secrets.")

# --- Global LLM and Embeddings Initialization ---
llm: Optional[ChatOpenAI] = None
embeddings_model: Optional[OpenAIEmbeddings] = None

if OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        print("DEBUG: OpenAI LLM and Embeddings models initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI models. Check API key validity/credits or internet connection: {e}")
        llm = None
        embeddings_model = None
else:
    # This block should ideally not be reached if the ValueError above is raised
    print("ERROR: Cannot initialize OpenAI models because OPENAI_API_KEY is missing (should have been caught by ValueError).")


def ingest_and_get_retriever() -> Optional[Chroma]:
    """
    Handles data ingestion (loading, chunking, embedding) and returns a ChromaDB instance.
    Loads from PDF and optionally from scraped data.
    Checks if the vector store already exists to avoid re-ingestion.
    """
    print("DEBUG: Entering ingest_and_get_retriever function.")

    if embeddings_model is None:
        print("ERROR: Embeddings model not initialized. Cannot ingest data or get retriever.")
        return None

    is_chroma_populated = False
    # Check if the persist directory exists and contains expected ChromaDB files
    if Path(chroma_dir_str).exists():
        # Specific check for ChromaDB's internal files (chroma.sqlite3 for old, collections for new)
        if (Path(chroma_dir_str) / "chroma.sqlite3").exists() or \
           (Path(chroma_dir_str) / "collections").exists():
            is_chroma_populated = True

    if is_chroma_populated:
        print(f"DEBUG: ChromaDB directory exists and seems populated at '{chroma_dir_str}'. Attempting to load existing DB.")
        try:
            vector_store = Chroma(persist_directory=chroma_dir_str, embedding_function=embeddings_model)
            # Add a count check to ensure it's not an empty DB
            if vector_store._collection.count() > 0:
                print(f"DEBUG: Loaded existing ChromaDB with {vector_store._collection.count()} documents.")
                return vector_store
            else:
                print("DEBUG: Existing ChromaDB found but it's empty. Proceeding with re-ingestion.")
                is_chroma_populated = False # Force re-ingestion
        except Exception as e:
            print(f"ERROR: Failed to load existing ChromaDB from '{chroma_dir_str}': {e}. Attempting re-ingestion.")
            is_chroma_populated = False # Force re-ingestion

    if not is_chroma_populated:
        print(f"\n--- DEBUG: Ingesting data as ChromaDB not found, empty, or failed to load ---")
        all_documents: List[Document] = []

        # --- Load PDF Document ---
        if not Path(pdf_path_str).exists():
            print(f"ERROR: The PDF file '{pdf_path_str}' does not exist. Cannot ingest PDF data.")
        else:
            print(f"DEBUG: Loading PDF from: {pdf_path_str}")
            pdf_loader = PyPDFLoader(pdf_path_str)
            try:
                # Use list(pdf_loader.lazy_load()) to force loading all pages
                pdf_pages = list(pdf_loader.lazy_load())
                for page in pdf_pages:
                    if page.page_content.strip():
                        all_documents.append(page)
                    else:
                        print(f"DEBUG: Warning: Page {page.metadata.get('page', 'N/A')} of '{os.path.basename(pdf_path_str)}' is empty or contains only whitespace. Skipping.")
                print(f"DEBUG: Successfully loaded {len(pdf_pages)} pages from the PDF.")
            except Exception as e:
                print(f"ERROR: Failed to load pages from PDF '{pdf_path_str}': {e}")

        # --- Load Scraped Text Documents (Optional) ---
        if Path(scraped_data_dir_str).exists():
            print(f"DEBUG: Loading scraped data from: {scraped_data_dir_str}")
            try:
                # Use TextLoader for .txt files
                txt_loader = DirectoryLoader(scraped_data_dir_str, glob="**/*.txt", loader_cls=TextLoader)
                scraped_docs = txt_loader.load()
                print(f"DEBUG: Loaded {len(scraped_docs)} documents from scraped data.")
                all_documents.extend(scraped_docs)
            except Exception as e:
                print(f"ERROR: Failed to load scraped text documents from '{scraped_data_dir_str}': {e}")
        else:
            print(f"DEBUG: Scraped data directory '{scraped_data_dir_str}' not found. Skipping scraped data loading.")


        if not all_documents:
            print(f"ERROR: No documents were loaded from PDF or scraped data. Cannot proceed with ingestion.")
            return None

        print(f"DEBUG: Total documents loaded for ingestion: {len(all_documents)}.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        document_chunks: List[Document] = text_splitter.split_documents(all_documents)
        print(f"DEBUG: Documents chunked into {len(document_chunks)} pieces.")

        print(f"DEBUG: Creating ChromaDB vector store in '{chroma_dir_str}'...")
        os.makedirs(chroma_dir_str, exist_ok=True)
        try:
            vector_store = Chroma.from_documents(
                documents=document_chunks,
                embedding=embeddings_model,
                persist_directory=chroma_dir_str
            )
            print(f"DEBUG: ChromaDB vector store created and persisted with {vector_store._collection.count()} documents.")
            return vector_store
        except Exception as e:
            print(f"ERROR: Failed to create/persist ChromaDB vector store during ingestion: {e}")
            return None
    
    return None # Should only be reached if is_chroma_populated was true and loading failed, and then re-ingestion also failed.


# --- Main RAG Chain Function ---
def get_conversational_rag_chain():
    """
    Initializes and returns a conversational RAG chain with memory.
    """
    print("DEBUG: Entering get_conversational_rag_chain function.")

    # Ensure LLM and embeddings_model are initialized globally
    if llm is None or embeddings_model is None:
        print("ERROR: LLM or Embeddings model not initialized. Cannot create conversational RAG chain.")
        return None
    print("DEBUG: LLM and Embeddings models are available.")

    # Get or create the vector store
    vector_store = ingest_and_get_retriever()

    if vector_store is None:
        print("ERROR: Failed to get/create vector store. Cannot create conversational RAG chain.")
        return None
    print("DEBUG: Vector store obtained.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("DEBUG: Retriever created.")

    # 1. Contextualize question (with history)
    contextualize_q_system_prompt = """Given a chat history and the latest user question, \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if necessary and otherwise return it as is.
    If the question is completely irrelevant to the chat history, return the original question as is.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    print("DEBUG: contextualize_q_prompt created.")

    # This is the line where the NameError occurred. Added more explicit prints.
    print("DEBUG: Attempting to define contextualize_q_chain...")
    try:
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
        print("DEBUG: contextualize_q_chain successfully defined.")
    except Exception as e:
        print(f"ERROR: Failed to define contextualize_q_chain: {e}")
        return None # Return None if this critical step fails

    # Using LangChain's helper for history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    print("DEBUG: history_aware_retriever created.")

    # 2. Answer Generation Prompt (with history and context)
    qa_system_prompt = """You are an AI assistant specializing in the Pakistan Penal Code (PPC).
    Your task is to answer questions about the PPC based *only* on the following context and chat history.
    If the context and chat history do not contain the information to answer the question,
    you MUST state: "I am unable to find the answer to your question within the provided Pakistan Penal Code document."
    Do not make up any information. Provide concise and accurate answers, citing relevant sections if applicable.
    If the user asks a question not related to the Pakistan Penal Code, politely inform them that you are specialized in the PPC and cannot answer the question.

    Context: {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"), # Pass the chat history directly to the main QA prompt
            ("human", "{input}"),
        ]
    )
    print("DEBUG: qa_prompt created.")

    # 3. Document Stuffing Chain
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    print("DEBUG: document_chain created.")

    # 4. Combine into a retrieval chain
    # The context will be the output of history_aware_retriever
    # The answer will be the output of document_chain
    print("DEBUG: Attempting to define retrieval_chain...")
    try:
        retrieval_chain = RunnablePassthrough.assign(
            context=contextualize_q_chain | retriever # This is the line where the error occurred
        ).assign(
            answer=document_chain
        )
        print("DEBUG: retrieval_chain successfully defined.")
    except Exception as e:
        print(f"ERROR: Failed to define retrieval_chain: {e}")
        return None # Return None if this critical step fails


    # 5. Add memory (Crucial for conversational history management by LangChain)
    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        # This lambda tells LangChain how to get the chat history for a session_id
        # For Streamlit, st.session_state will manage the actual history.
        # This ChatMessageHistory is ephemeral per invocation for LangChain's internal use.
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input", # Key for current user input
        history_messages_key="chat_history", # Key for history passed to prompts
        output_messages_key="answer", # Key for the final answer
    )

    print("DEBUG: Conversational RAG chain constructed successfully.")
    return conversational_rag_chain


# --- Direct Testing Block (will only run if backend.py is executed directly) ---
if __name__ == "__main__":
    print("DEBUG: Testing backend.py directly with conversational chain...")
    # Ensure a dummy API key is set for local testing if not in .env
    # This block will only run if backend.py is executed directly, not when imported by app.py
    if os.getenv("OPENAI_API_KEY") is None:
        # NOTE: For actual local testing, replace "sk-..." with your real key
        os.environ["OPENAI_API_KEY"] = "sk-YOUR_TEST_OPENAI_API_KEY_HERE"
        print("DEBUG: Set dummy OPENAI_API_KEY for direct backend.py testing.")

    chain = get_conversational_rag_chain()
    if chain:
        session_id = "test_session_123" # A unique ID for the conversation session

        print("\n--- Turn 1 ---")
        query1 = "if i take a minor girl from one place to another. What does the Pakistan Penal Code say about this act?"
        print(f"DEBUG: Querying: '{query1}'")
        try:
            # For direct testing, you need to explicitly pass an empty chat_history for the first turn
            response1 = chain.invoke({"input": query1, "chat_history": []}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response1.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Turn 1): {e}")

        print("\n--- Turn 2 ---")
            # LangChain's RunnableWithMessageHistory will manage the history internally
            # based on session_id, so you don't need to pass chat_history explicitly here
        query2 = "What will be the punishment for this?" # Follow-up question
        print(f"DEBUG: Querying: '{query2}'")
        try:
            response2 = chain.invoke({"input": query2}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response2.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Turn 2): {e}")

    else:
        print("ERROR: Failed to get conversational RAG chain.")