import os
import pickle
from dotenv import load_dotenv

# --- LlamaParse & LangChain Core ---
from llama_parse import LlamaParse
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

# --- Embeddings ---
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings 

# --- Vector Store & Retrievers ---
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

load_dotenv()

COLLECTION_NAME = "zendesk_support_hybrid"
EMBEDDING_MODEL_NAME = "nvidia/llama-nemotron-embed-1b-v2"

# --- ABSOLUTE PATHS ---
BASE_DIR = r"C:/VOICE_AI_AGENT"
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
BM25_SAVE_PATH = os.path.join(BASE_DIR, "bm25_retriever.pkl")

def parse_pdfs_with_llamaparse(pdf_directory: str):
    """
    Sends PDFs to LlamaParse API to extract highly accurate Markdown,
    preserving tables and document structure from support manuals.
    """
    print(f"📂 Parsing PDFs from '{pdf_directory}' using LlamaParse API...")
    raw_langchain_docs = []
    
    if not os.path.exists(pdf_directory):
        print(f"⚠️ Directory '{pdf_directory}' does not exist.")
        return raw_langchain_docs

    llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_api_key:
        raise ValueError("❌ LLAMA_CLOUD_API_KEY not found in your .env file!")

    # Initialize LlamaParse to output Markdown (perfect for tables/structure)
    parser = LlamaParse(
        api_key=llama_api_key,
        result_type="markdown",
        verbose=True
    )

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            print(f"   📄 Parsing and structuring: {filename} (via LlamaParse)")
            
            try:
                # LlamaParse returns LlamaIndex documents
                llama_docs = parser.load_data(filepath)
                
                # Combine pages into a single markdown text for the file
                full_markdown_text = "\n\n".join([doc.text for doc in llama_docs])
                
                # Convert to a standard LangChain Document
                langchain_doc = Document(
                    page_content=full_markdown_text,
                    metadata={"source": filename}
                )
                raw_langchain_docs.append(langchain_doc)
                
            except Exception as e:
                print(f"   ❌ Error parsing {filename}: {e}")
            
    print(f"✅ Extracted structured markdown from {len(raw_langchain_docs)} support documents.")
    return raw_langchain_docs

def preprocess_and_index(pdf_directory: str):
    print("--- STARTING KNOWLEDGE BASE GENERATION ---")
    
    # 1. Structural Parsing via LlamaParse
    raw_docs = parse_pdfs_with_llamaparse(pdf_directory)
    
    if not raw_docs:
        print("❌ No documents found. Exiting preprocessing.")
        return

    # 2. Chunking the Markdown
    print("✂️ Chunking markdown text while preserving structure...")
    splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs = splitter.split_documents(raw_docs)
    print(f"✅ Generated {len(docs)} structural chunks.")

    # 3. Initialize NVIDIA Embeddings
    print(f"🧠 Initializing NVIDIA Embeddings ({EMBEDDING_MODEL_NAME})...")
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("❌ NVIDIA_API_KEY not found in your .env file!")
        
    embedding_model = NVIDIAEmbeddings(
        model=EMBEDDING_MODEL_NAME
    )

    # 4. Save to Vector Store (ChromaDB)
    print("💾 Initializing and Persisting ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embedding_model, 
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print(f"✅ ChromaDB persisted to '{CHROMA_PERSIST_DIR}'")

    # 5. Save to Sparse Retriever (BM25)
    print("💾 Initializing and Saving BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    
    with open(BM25_SAVE_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"✅ BM25 Retriever saved to '{BM25_SAVE_PATH}'")
    print("🎉 Knowledge Base Preprocessing Complete!")

if __name__ == "__main__":
    TARGET_PDF_DIR = os.path.join(BASE_DIR, "data")
    
    if not os.path.exists(TARGET_PDF_DIR):
        print(f"⚠️ Critical Error: The path '{TARGET_PDF_DIR}' does not exist.")
        print(f"Please create the folder and place your Zendesk PDF inside it.")
    else:
        preprocess_and_index(TARGET_PDF_DIR)