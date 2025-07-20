import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Optional
import streamlit as st
from pathlib import Path
import requests # Used for downloading
import zipfile # Used for unzipping
import io # Used for in-memory zip handling
import shutil # Used for removing directories
# Removed subprocess as wget is no longer used

# --- IMPORTANT: Workaround for ChromaDB SQLite3 issue on Streamlit Cloud ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("DEBUG: Successfully swapped sqlite3 with pysqlite3.")
except ImportError:
    print("DEBUG: pysqlite3 not found or import failed. Using default sqlite3.")
    pass


# LangChain components for RAG
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain


# Load environment variables (important for local development)
load_dotenv()

# --- Configuration for Paths ---
current_file_dir = Path(__file__).parent
PROJECT_ROOT_DIR = current_file_dir.parent

PPC_PDF_DIR = PROJECT_ROOT_DIR / "data"
SCRAPED_DATA_DIR = PROJECT_ROOT_DIR / "scraped_legal_data"

ppc_pdf_dir_str = str(PPC_PDF_DIR)
scraped_data_dir_str = str(SCRAPED_DATA_DIR)

CHROMA_PERSIST_DIRECTORY = PROJECT_ROOT_DIR / "db"
chroma_dir_str = str(CHROMA_PERSIST_DIRECTORY)

# --- IMPORTANT: Configure your pre-built DB download URL here ---
# This URL should be a direct download link for your db.zip file.
# For Google Drive, use the format: https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE
# The 'confirm=t' parameter is often helpful for large files to bypass warning pages.
PREBUILT_DB_URL = "https://github.com/saudch007/PPC-Legal-AI/releases/download/data-v1.0.0-t1/db.zip" # <--- REPLACE 'YOUR_FILE_ID_HERE' WITH YOUR ACTUAL GOOGLE DRIVE FILE ID

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

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
    Prioritizes loading from an existing vector store or downloading a pre-built one.
    Falls back to fresh ingestion if other methods fail.
    """
    print("DEBUG: Entering ingest_and_get_retriever function.")

    if embeddings_model is None:
        print("ERROR: Embeddings model not initialized. Cannot ingest data or get retriever.")
        return None

    # 1. Attempt to load existing local ChromaDB
    is_chroma_populated_locally = False
    if Path(chroma_dir_str).exists():
        if (Path(chroma_dir_str) / "chroma.sqlite3").exists() or \
           (Path(chroma_dir_str) / "collections").exists():
            try:
                vector_store = Chroma(persist_directory=chroma_dir_str, embedding_function=embeddings_model)
                if vector_store._collection.count() > 0:
                    is_chroma_populated_locally = True
                    print(f"DEBUG: Loaded existing ChromaDB with {vector_store._collection.count()} documents.")
                    return vector_store
                else:
                    print("DEBUG: Existing ChromaDB found but it's empty. Proceeding with other methods.")
            except Exception as e:
                print(f"ERROR: Failed to load existing ChromaDB from '{chroma_dir_str}': {e}. Attempting other methods.")

    # 2. If local DB is not populated, attempt to download pre-built DB
    # Check if PREBUILT_DB_URL has been configured (i.e., not the placeholder)
    if not is_chroma_populated_locally and PREBUILT_DB_URL == "https://github.com/saudch007/PPC-Legal-AI/releases/download/data-v1.0.0-t1/db.zip":
        print(f"\n--- DEBUG: Local ChromaDB not populated. Attempting to download pre-built DB from {PREBUILT_DB_URL} using requests ---")
        try:
            # Clean up any incomplete/corrupt local db folder before downloading
            if Path(chroma_dir_str).exists():
                print(f"DEBUG: Clearing existing '{chroma_dir_str}' before downloading pre-built DB.")
                shutil.rmtree(chroma_dir_str)
            os.makedirs(chroma_dir_str, exist_ok=True)

            # Use requests.get with stream=True for large files and allow_redirects
            response = requests.get(PREBUILT_DB_URL, stream=True, allow_redirects=True, timeout=300) # Increased timeout
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            print(f"DEBUG: Download response status code: {response.status_code}")
            print(f"DEBUG: Download response headers: {response.headers}") # Print headers for debugging

            # Check if the response is actually a file download (not an HTML page)
            content_type = response.headers.get('Content-Type', '')
            content_disposition = response.headers.get('Content-Disposition', '')

            # If it's not a direct attachment or is HTML, it might be an HTML page (e.g., warning page)
            if 'application/zip' not in content_type and 'application/octet-stream' not in content_type and 'filename=' not in content_disposition:
                # Try to read a small part of content to check if it's HTML
                # Use a BytesIO buffer to store the initial chunk so it can be re-read by zipfile
                buffer = io.BytesIO()
                initial_content_chunk = next(response.iter_content(chunk_size=1024))
                buffer.write(initial_content_chunk)
                buffer.seek(0) # Rewind buffer to read from beginning

                initial_content_str = buffer.read().decode('utf-8', errors='ignore')
                buffer.seek(0) # Rewind again for zipfile

                if "<!DOCTYPE html" in initial_content_str.lower() or "<html" in initial_content_str.lower():
                    raise requests.exceptions.RequestException("Response is likely an HTML page, not a direct file download. Check Google Drive sharing permissions or URL.")
                
                # If it's not HTML, the buffer already contains the initial chunk
                file_content_stream = buffer
            else:
                file_content_stream = io.BytesIO()

            # Write the rest of the stream content
            for chunk in response.iter_content(chunk_size=8192):
                file_content_stream.write(chunk)
            file_content_stream.seek(0) # Rewind to the beginning for zipfile

            print(f"DEBUG: Downloaded stream. Attempting to extract zip from memory...")
            with zipfile.ZipFile(file_content_stream, 'r') as zip_ref:
                zip_ref.extractall(chroma_dir_str)

            print(f"DEBUG: Successfully downloaded and extracted pre-built DB to {chroma_dir_str}")

            # Verify the downloaded DB
            vector_store = Chroma(persist_directory=chroma_dir_str, embedding_function=embeddings_model)
            if vector_store._collection.count() > 0:
                print(f"DEBUG: Successfully loaded downloaded ChromaDB with {vector_store._collection.count()} documents.")
                return vector_store
            else:
                print("DEBUG: Downloaded DB is empty or corrupt. Falling back to fresh ingestion.")

        except requests.exceptions.RequestException as req_err:
            print(f"ERROR: Network/Request error downloading pre-built DB: {req_err}. Falling back to fresh ingestion.")
        except zipfile.BadZipFile as zip_err:
            print(f"ERROR: Downloaded file is not a valid zip: {zip_err}. This often means the URL provided an HTML page instead of a zip file. Falling back to fresh ingestion.")
        except Exception as e:
            print(f"ERROR: Failed to download or extract pre-built DB using requests: {e}. Falling back to fresh ingestion.")
    elif not is_chroma_populated_locally and PREBUILT_DB_URL == "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE&confirm=t":
        print("DEBUG: PREBUILT_DB_URL not configured. Proceeding with fresh ingestion.")

    # 3. Fallback to Fresh Ingestion (if download fails or not configured)
    print(f"\n--- DEBUG: Performing fresh data ingestion ---")
    all_documents: List[Document] = []

    # --- Load ALL PDF Documents from PPC_PDF_DIR (e.g., 'data' folder) ---
    if not Path(ppc_pdf_dir_str).exists():
        print(f"ERROR: The primary PDF directory '{ppc_pdf_dir_str}' does not exist. Cannot ingest from here.")
    else:
        print(f"DEBUG: Loading PDFs from: {ppc_pdf_dir_str} (recursively)")
        try:
            pdf_loader_primary = PyPDFDirectoryLoader(ppc_pdf_dir_str)
            primary_pdf_docs = pdf_loader_primary.load()
            print(f"DEBUG: Found {len(primary_pdf_docs)} PDF documents in '{ppc_pdf_dir_str}'.")
            for doc in primary_pdf_docs:
                if doc.page_content.strip():
                    all_documents.append(doc)
                    print(f"DEBUG: Loaded PDF: {doc.metadata.get('source', 'Unknown Source')}, Page {doc.metadata.get('page', 'N/A')}. Content snippet: '{doc.page_content[:200].replace('\n', ' ')}...'")
                else:
                    print(f"DEBUG: Warning: Empty page content found in {doc.metadata.get('source', 'N/A')} page {doc.metadata.get('page', 'N/A')}. Skipping.")
            print(f"DEBUG: Successfully loaded {len(primary_pdf_docs)} documents from primary PDF directory.")
        except Exception as e:
            print(f"ERROR: Failed to load PDFs from '{ppc_pdf_dir_str}': {e}")

    # --- Load Scraped Text Documents from SCRAPED_DATA_DIR ---
    if not Path(scraped_data_dir_str).exists():
        print(f"DEBUG: Scraped data directory '{scraped_data_dir_str}' not found. Skipping scraped data loading.")
    else:
        print(f"DEBUG: Loading scraped text data from: {scraped_data_dir_str}")
        scraped_docs_list: List[Document] = []
        for file_entry in os.listdir(scraped_data_dir_str):
            file_path = os.path.join(scraped_data_dir_str, file_entry)
            if os.path.isfile(file_path) and file_path.lower().endswith('.txt'):
                try:
                    single_txt_loader = TextLoader(file_path, encoding='utf-8')
                    loaded_doc = single_txt_loader.load()
                    if loaded_doc and loaded_doc[0].page_content.strip():
                        scraped_docs_list.extend(loaded_doc)
                        print(f"DEBUG: Loaded Text: {file_path}. Content snippet: '{loaded_doc[0].page_content[:200].replace('\n', ' ')}...'")
                    else:
                        print(f"DEBUG: Warning: Empty page content found in {file_path}. Skipping.")
                except Exception as e:
                    print(f"ERROR: Failed to load text file {file_path}: {e}")
        
        print(f"DEBUG: Found {len(scraped_docs_list)} text documents in '{scraped_data_dir_str}'.")
        if scraped_docs_list:
            all_documents.extend(scraped_docs_list)
            print(f"DEBUG: Successfully loaded {len(scraped_docs_list)} documents from scraped text data.")
        else:
            print("DEBUG: No text documents were successfully loaded from scraped data directory.")


    if not all_documents:
        print(f"ERROR: No documents were loaded from any source. Cannot proceed with ingestion.")
        return None

    print(f"DEBUG: Total documents loaded for ingestion: {len(all_documents)}.")
    print(f"DEBUG: Total documents (PDFs + Text) collected before chunking: {len(all_documents)}.")


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

    # Modified: Increased k to 20 for even broader retrieval
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    print("DEBUG: Retriever created.")

    # 1. Query Rewriter Prompt (for standalone question generation)
    query_rewriter_system_prompt = """Given the following conversation history and the latest user input (which might be a question or case facts), \
    your primary task is to rephrase the latest input into a clear, concise, and standalone query. \
    This rephrased query must be fully understandable without any reference to the previous chat history. \
    It will be used to retrieve relevant legal documents.

    Key instructions for rephrasing:
    - **Resolve all pronouns:** Replace any pronouns (e.g., "it", "its", "he", "she", "this", "that") with the specific noun or entity they refer to from the conversation history.
    - **Incorporate full context:** Ensure the rephrased query contains all necessary context from previous turns to stand alone. For example, if the history is "What is dacoity?" and the new question is "What is its punishment?", the rephrased question should be "What is the punishment for dacoity?".
    - **Be specific:** Add details from the conversation history to make the query precise.
    - **Handle meta-questions directly:** If the user's input is about the conversation itself (e.g., "what was my previous question?", "who are you?", "can you remember this?"), \
      do NOT rephrase it for document retrieval. Instead, return the original meta-question exactly as it was asked. The main answer generation LLM will handle these.
    - **Output format:** The output must be ONLY the rephrased query string. Do not include any conversational filler, introductory phrases, or punctuation beyond what is necessary for a clear query. Do not answer the query.

    Example 1:
    Chat History:
    Human: What is theft?
    AI: Theft is defined as...
    Human: What is its punishment?
    Rephrased Query: What is the punishment for theft?

    Example 2:
    Chat History:
    Human: Tell me about Section 302.
    AI: Section 302 deals with punishment for murder.
    AI: Section 302 deals with punishment for murder.
    Human: What about Section 303?
    Rephrased Query: What about Section 303 of the Pakistan Penal Code?

    Example 3:
    Chat History:
    Human: What is dacoity?
    AI: Dacoity is defined under Section 391...
    Human: what was my previous question?
    Rephrased Query: what was my previous question?
    """
    query_rewriter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", query_rewriter_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    print("DEBUG: query_rewriter_prompt created.")

    print("DEBUG: Attempting to define query_rewriter_chain...")
    try:
        query_rewriter_chain = query_rewriter_prompt | llm | StrOutputParser()
        print("DEBUG: query_rewriter_chain successfully defined.")
    except Exception as e:
        print(f"ERROR: Failed to define query_rewriter_chain: {e}")
        return None

    # 2. Answer Generation Prompt (for judicial analysis)
    qa_system_prompt = """You are an AI Legal Analyst, providing a second opinion based *only* on the provided **Pakistani legal documents**, which include the Pakistan Penal Code (PPC), Supreme Court of Pakistan Judgments, and other relevant legal texts.
    Your task is to analyze the given case information or legal query, identify relevant legal principles, sections, and precedents from the provided documents, and explain their potential applicability.

    Key instructions:
    - **Role:** You are an assistant to a judge, providing legal analysis, not making a binding judgment.
    - **Strictly adhere to provided documents:** Your analysis must be based *only* on the content of the documents you have access to.
    - **Acknowledge Data Gaps (Crucial):** If the retrieved context does not contain sufficient information to provide a comprehensive analysis for a specific legal area (e.g., if foundational statutes like Family Laws or Civil Procedure Code are not present, but only judgments referring to them are), explicitly state this limitation. For example, you might say: "Based on the provided documents, I can analyze aspects related to the Pakistan Penal Code and available judgments. However, a comprehensive analysis of [e.g., Family Law aspects like Khula or Dowry] would require the full text of relevant statutes such as the [e.g., Dissolution of Muslim Marriages Act, Dowry and Bridal Gifts (Restriction) Act, Family Courts Act], which are not present in my current knowledge base."
    - **Prioritize Relevant Findings:** Even if a comprehensive analysis isn't possible, if you find *any* relevant judgments or PPC sections that touch upon aspects of the query, you MUST present those findings clearly, before stating any limitations. Do not give a blanket "unable to provide" if partial information exists.
    - **No Hallucinations:** Do not invent or infer any information not explicitly present in the provided context.
    - **Concise and Accurate:** Provide answers that are to the point and factually correct based on the provided legal documents.
    - **Cite Sources:** Always cite the relevant section numbers from the PPC (e.g., "Section 391 states...") or reference specific judgments if the information is derived from them (e.g., "As per the judgment in [Case Name/Citation]...").
    - **Formal and Objective Tone:** Maintain a professional, objective, and formal tone. Avoid colloquialisms, emojis, or excessive punctuation.
    - **Legal Scope Only:** If the input is clearly outside the scope of Pakistani legal documents (e.g., general knowledge, personal opinions, or questions about other legal systems), \
      politely inform the user that you are specialized in Pakistani law and cannot analyze the given information. Do not attempt to answer unrelated queries.
    - **Disclaimer:** Conclude every analysis with a clear disclaimer: "Please note: This is an AI-generated legal analysis based solely on the provided Pakistani legal documents. It is for informational purposes only and does not constitute legal advice or a judicial ruling. A qualified legal professional should always be consulted for definitive legal opinions and decisions."

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

    # 4. Define the overall RAG chain flow
    print("DEBUG: Attempting to define full_rag_chain_core...")
    try:
        full_rag_chain_core = (
            RunnablePassthrough.assign(
                standalone_question=query_rewriter_chain
            )
            | RunnablePassthrough.assign(
                context=lambda x: retriever.invoke(x["standalone_question"])
            )
            | document_chain
        )
        print("DEBUG: full_rag_chain_core successfully defined.")
    except Exception as e:
            print(f"ERROR: Failed to define full_rag_chain_core: {e}")
            return None

    # 5. Add memory (Crucial for conversational history management by LangChain)
    conversational_rag_chain = RunnableWithMessageHistory(
        full_rag_chain_core, # Use the new core chain
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer", # The document_chain outputs the answer
    )

    print("DEBUG: Conversational RAG chain constructed successfully.")
    return conversational_rag_chain


# --- Direct Testing Block (will only run if backend.py is executed directly) ---
if __name__ == "__main__":
    print("DEBUG: Testing backend.py directly with conversational chain...")
    if os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = "sk-YOUR_TEST_OPENAI_API_KEY_HERE"
        print("DEBUG: Set dummy OPENAI_API_KEY for direct backend.py testing.")

    chain = get_conversational_rag_chain()
    if chain:
        session_id = "test_session_123"

        if session_id in ChatMessageHistory.store:
             del ChatMessageHistory.store[session_id]

        print("\n--- Test Case 1: Dacoity and its punishment ---")
        query1 = "A group of six individuals, armed with knives, entered a bank and forcibly took money from the cashier. They threatened customers but did not physically harm anyone. What does the Pakistan Penal Code say about this act, and what would be the rightful decision?"
        print(f"DEBUG: Querying: '{query1}'")
        try:
            response1 = chain.invoke({"input": query1, "chat_history": []}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response1.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Test Case 1): {e}")

        print("\n--- Test Case 2: Follow-up on punishment ---")
        query2 = "What punishment is prescribed for such an act?"
        print(f"DEBUG: Querying: '{query2}'")
        try:
            response2 = chain.invoke({"input": query2}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response2.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Test Case 2): {e}")

        print("\n--- Test Case 3: Kidnapping scenario ---")
        query3 = "A person secretly takes a 15-year-old girl from her school without her parents' consent, intending to marry her. What sections of the PPC are relevant here, and what is the legal implication?"
        print(f"DEBUG: Querying: '{query3}'")
        try:
            response3 = chain.invoke({"input": query3}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response3.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Test Case 3): {e}")

        print("\n--- Test Case 4: Irrelevant question ---")
        query4 = "What is the capital of France?"
        print(f"DEBUG: Querying: '{query4}'")
        try:
            response4 = chain.invoke({"input": query4}, config={"configurable": {"session_id": session_id}})
            print(f"Answer: {response4.get('answer', 'No answer found.')}")
        except Exception as e:
            print(f"ERROR: During RAG chain invocation (Test Case 4): {e}")

    else:
        print("ERROR: Failed to get conversational RAG chain.")
