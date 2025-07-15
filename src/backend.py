import os
import sys # Import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Optional
import streamlit as st
from pathlib import Path

# --- IMPORTANT: Workaround for ChromaDB SQLite3 issue on Streamlit Cloud ---
# This forces ChromaDB to use the pysqlite3-binary package if available,
# which provides a newer SQLite3 version compatible with ChromaDB.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("DEBUG: Successfully swapped sqlite3 with pysqlite3.")
except ImportError:
    print("DEBUG: pysqlite3 not found or import failed. Using default sqlite3.")
    pass # Fallback to default sqlite3 if pysqlite3 isn't available/working


# LangChain components for RAG
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma # Corrected import
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever


# Load environment variables (important for local development)
load_dotenv()

# --- Configuration for Paths ---
current_file_dir = Path(__file__).parent
PROJECT_ROOT_DIR = current_file_dir.parent

PDF_FILE_PATH = PROJECT_ROOT_DIR / "data" / "Pakistan_Penal_Code.pdf"
CHROMA_PERSIST_DIRECTORY = PROJECT_ROOT_DIR / "db"
SCRAPED_DATA_DIR = PROJECT_ROOT_DIR / "scraped_legal_data" # Placeholder for future scraping

pdf_path_str = str(PDF_FILE_PATH)
chroma_dir_str = str(CHROMA_PERSIST_DIRECTORY)
scraped_data_dir_str = str(SCRAPED_DATA_DIR)


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- Corrected OpenAI API Key Handling (Global Initialization) ---
OPENAI_API_KEY: Optional[str] = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print("DEBUG: Loaded OPENAI_API_KEY from local .env file.")
else:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        print("DEBUG: Loaded OPENAI_API_KEY from Streamlit secrets.")
    else:
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
    if Path(chroma_dir_str).exists():
        if (Path(chroma_dir_str) / "chroma.sqlite3").exists() or \
           (Path(chroma_dir_str) / "collections").exists():
            is_chroma_populated = True

    if is_chroma_populated:
        print(f"DEBUG: ChromaDB directory exists and seems populated at '{chroma_dir_str}'. Attempting to load existing DB.")
        try:
            vector_store = Chroma(persist_directory=chroma_dir_str, embedding_function=embeddings_model)
            if vector_store._collection.count() > 0:
                print(f"DEBUG: Loaded existing ChromaDB with {vector_store._collection.count()} documents.")
                return vector_store
            else:
                print("DEBUG: Existing ChromaDB found but it's empty. Proceeding with re-ingestion.")
                is_chroma_populated = False
        except Exception as e:
            print(f"ERROR: Failed to load existing ChromaDB from '{chroma_dir_str}': {e}. Attempting re-ingestion.")
            is_chroma_populated = False

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
    
    return None


# --- Main RAG Chain Function ---
def get_conversational_rag_chain():
    """
    Initializes and returns a conversational RAG chain with memory.
    """
    print("DEBUG: Entering get_conversational_rag_chain function.")

    if llm is None or embeddings_model is None:
        print("ERROR: LLM or Embeddings model not initialized. Cannot create conversational RAG chain.")
        return None
    print("DEBUG: LLM and Embeddings models are available.")

    vector_store = ingest_and_get_retriever()

    if vector_store is None:
        print("ERROR: Failed to get/create vector store. Cannot create conversational RAG chain.")
        return None
    print("DEBUG: Vector store obtained.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("DEBUG: Retriever created.")

    # 1. Contextualize question (with history)
    # IMPROVED: Make the prompt more explicit about resolving pronouns and follow-ups.
    contextualize_q_system_prompt = """Given the following conversation history and the latest user question, \
    your task is to rephrase the follow-up question into a clear, standalone question that can be understood \
    without referring back to the chat history. This rephrased question will be used to retrieve relevant documents.

    Key instructions:
    - **Resolve Pronouns:** If the follow-up question contains pronouns (e.g., "its", "he", "she", "this"), \
      replace them with the specific entity from the previous conversation turn.
    - **Incorporate Context:** Ensure the rephrased question includes enough context from the chat history \
      so it stands alone. For example, if the previous question was "What is dacoity?" and the follow-up is "What is its punishment?", \
      the rephrased question should be "What is the punishment for dacoity?".
    - **Handle Meta-Questions:** If the question is a meta-question about the conversation itself (e.g., "what was my previous question?", "who are you?"), \
      do not attempt to rephrase it for document retrieval. Instead, return the original meta-question as is. The main LLM will handle these.
    - **Do NOT Answer:** Your sole purpose is to rephrase the question. Do not provide an answer to the question.
    - **Be Direct:** The output should be a direct, rephrased question, without conversational filler.

    Example:
    Chat History:
    Human: What is theft?
    AI: Theft is defined as...
    Human: What is its punishment?
    Rephrased Question: What is the punishment for theft?

    Chat History:
    Human: Tell me about Section 302.
    AI: Section 302 deals with punishment for murder.
    Human: What about Section 303?
    Rephrased Question: What about Section 303 of the Pakistan Penal Code?

    Chat History:
    Human: What is dacoity?
    AI: Dacoity is defined under Section 391...
    Human: what was my previous question?
    Rephrased Question: what was my previous question?
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    print("DEBUG: contextualize_q_prompt created.")

    print("DEBUG: Attempting to define contextualize_q_chain...")
    try:
        # The history-aware retriever will use this chain to rewrite the query.
        # This is where the magic for follow-up questions happens.
        contextualize_q_chain = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        print("DEBUG: contextualize_q_chain (history-aware retriever) successfully defined.")
    except Exception as e:
        print(f"ERROR: Failed to define contextualize_q_chain: {e}")
        return None

    # 2. Answer Generation Prompt (with history and context)
    # IMPROVED: Emphasize formal, direct answers and clarity on non-PPC questions.
    qa_system_prompt = """You are a highly specialized AI assistant focused exclusively on the Pakistan Penal Code (PPC).
    Your primary task is to provide accurate, formal, and direct answers to user questions based *only* on the following retrieved context and the provided chat history.
    The chat history is essential for understanding the full context of the user's current question, especially for follow-up inquiries.

    Key instructions:
    - **Strictly adhere to context:** If the retrieved context and chat history do NOT contain sufficient information to answer the question, \
      you MUST respond with: "I am unable to find the answer to your question within the provided Pakistan Penal Code document."
    - **No Hallucinations:** Do not invent or infer any information not explicitly present in the provided context.
    - **Concise and Accurate:** Provide answers that are to the point and factually correct based on the PPC.
    - **Cite Sections (if applicable):** If an answer is directly from a specific section, mention the section number (e.g., "Section 391 states...").
    - **PPC Scope Only:** If the user asks a question that is clearly outside the scope of the Pakistan Penal Code (e.g., general knowledge, personal opinions, or questions about other legal systems), \
      politely inform them that you are specialized in the PPC and cannot answer the question. Do not attempt to answer unrelated questions.
    - **Formal Tone:** Maintain a professional and formal tone. Avoid colloquialisms, emojis, or excessive punctuation (like multiple exclamation marks or question marks).

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
    print("DEBUG: Attempting to define conversational_rag_chain's core logic...")
    try:
        # This is the main chain that orchestrates history-aware retrieval and then answers
        # The 'contextualize_q_chain' (history-aware retriever) takes the original input and chat_history
        # and outputs a rewritten query. This rewritten query is what 'retriever.get_relevant_documents'
        # then uses to fetch context.
        conversational_rag_chain_core = contextualize_q_chain | RunnablePassthrough.assign(
            context=lambda x: retriever.get_relevant_documents(x["input"]) # x["input"] here is the rewritten query from contextualize_q_chain
        ) | document_chain
        print("DEBUG: conversational_rag_chain_core successfully defined.")
    except Exception as e:
        print(f"ERROR: Failed to define conversational_rag_chain_core: {e}")
        return None


    # 5. Add memory (Crucial for conversational history management by LangChain)
    # The RunnableWithMessageHistory wrapper manages the chat_history for the entire chain.
    conversational_rag_chain = RunnableWithMessageHistory(
        conversational_rag_chain_core, # Use the new core chain
        # This lambda tells LangChain how to get the chat history for a session_id.
        # For Streamlit, st.session_state will manage the actual history.
        # This ChatMessageHistory is ephemeral per invocation for LangChain's internal use.
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input", # Key for current user input
        history_messages_key="chat_history", # Key for history passed to prompts
        # The output_messages_key is for the final answer from the chain
        output_messages_key="answer",
    )

    print("DEBUG: Conversational RAG chain constructed successfully.")
    return conversational_rag_chain


# --- Direct Testing Block (will only run if backend.py is executed directly) ---
if __name__ == "__main__":
    print("DEBUG: Testing backend.py directly with conversational chain...")
    if os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = "sk-YOUR_TEST_OPENAI_API_KEY_HERE" # Replace with your real key for local testing
        print("DEBUG: Set dummy OPENAI_API_KEY for direct backend.py testing.")

    chain = get_conversational_rag_chain()
    if chain:
        session_id = "test_session_123" # A unique ID for the conversation session

        print("\n--- Turn 1 ---")
        query1 = "what is dacoity"
        print(f"DEBUG: Querying: '{query1}'")
        try:
            # For direct testing, you need to explicitly pass an empty chat_history for the first turn
            response1 = chain.invoke({"input": query1, "chat_history": []}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response1.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Turn 1): {e}")

        print("\n--- Turn 2 ---")
        query2 = "what is its punishment" # Follow-up question
        print(f"DEBUG: Querying: '{query2}'")
        try:
            # LangChain's RunnableWithMessageHistory will manage the history internally
            # based on session_id, so you don't need to pass chat_history explicitly here
            response2 = chain.invoke({"input": query2}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response2.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Turn 2): {e}")

        print("\n--- Turn 3 ---")
        query3 = "what was my previous question?" # Meta-question
        print(f"DEBUG: Querying: '{query3}')") # Added closing parenthesis for consistency
        try:
            response3 = chain.invoke({"input": query3}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response3.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Turn 3): {e}")

        print("\n--- Turn 4 ---")
        query4 = "what is kidnapping?" # New question, testing context reset
        print(f"DEBUG: Querying: '{query4}'")
        try:
            response4 = chain.invoke({"input": query4}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response4.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Turn 4): {e}")

        print("\n--- Turn 5 ---")
        query5 = "punishment for it" # Follow-up to kidnapping
        print(f"DEBUG: Querying: '{query5}'")
        try:
            response5 = chain.invoke({"input": query5}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response5.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Turn 5): {e}")

    else:
        print("ERROR: Failed to get conversational RAG chain.")
