import os
import pickle
from dotenv import load_dotenv

# Embeddings & Vector Stores
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma

# Sparse Retriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# LLM & Prompts
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Langfuse for tracing
from langfuse.langchain import CallbackHandler
from langfuse import get_client

# --- IMPORT YOUR PREPROCESSOR ---
from rag_preprocess import preprocess_and_index

load_dotenv()

# --- CONSTANTS & PATHS ---
COLLECTION_NAME = "zendesk_support_hybrid"
EMBEDDING_MODEL_NAME = "nvidia/llama-nemotron-embed-1b-v2" 
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Update BASE_DIR to match your current project structure
BASE_DIR = r"C:/VOICE_AI_AGENT"
TARGET_PDF_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
BM25_SAVE_PATH = os.path.join(BASE_DIR, "bm25_retriever.pkl")


class ZendeskRAGPipeline:
    def __init__(self, chroma_retriever, bm25_retriever, cross_encoder, prompt, llm):
        self.chroma_retriever = chroma_retriever
        self.bm25_retriever = bm25_retriever
        self.cross_encoder = cross_encoder 
        self.prompt = prompt
        self.llm = llm
        self.langfuse_client = get_client()

    def retrieve_and_rerank(self, query, top_k=5):
        print(f"🔍 Searching Support Knowledge Base for: '{query}'")
        dense_docs = self.chroma_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)
        
        # Deduplicate the results
        unique_docs = {}
        for doc in dense_docs + sparse_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
                
        doc_list = list(unique_docs.values())
        if not doc_list:
            return ""
            
        print(f"⚖️ Reranking {len(doc_list)} support snippets...")
        
        # 1. Create query-document pairs
        pairs = [[query, doc.page_content] for doc in doc_list]
        
        # 2. Predict the relevance scores
        scores = self.cross_encoder.predict(pairs)
        
        # 3. Zip documents with their scores and sort from highest to lowest
        scored_docs = list(zip(doc_list, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Extract the top_k results
        top_docs = [f"[Source {i+1}]: {doc.page_content}" for i, (doc, score) in enumerate(scored_docs[:top_k])]
        
        return "\n\n---\n\n".join(top_docs)

    def invoke(self, state: dict):
        original_question = state.get("original_question", "")
        # The chat history is passed in from memory (no DB needed)
        chat_history = state.get("chat_history", [])
        session_id = state.get("session_id", "anonymous_session")
        
        langfuse_handler = CallbackHandler()
        langfuse_config = {
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_tags": ["support_rag_query"]
            }
        }
        
        if chat_history:
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
            rewrite_sys = """Given the following customer conversation history and a follow-up query, rephrase the follow-up question to be a standalone search query.
            Do NOT answer the question, just reformulate it so it contains all the necessary context. If it's already standalone, return it exactly as is."""
            rewrite_prompt = PromptTemplate.from_template(f"{rewrite_sys}\n\nChat History:\n{{history}}\n\nFollow-Up: {{question}}\nStandalone Search Query:")
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            
            search_query = rewrite_chain.invoke(
                {"history": history_str, "question": original_question},
                config=langfuse_config
            )
            print(f"🔄 Rewrote Query for Context: {search_query}")
        else:
            search_query = original_question
            history_str = "No previous history."
            
        context_str = self.retrieve_and_rerank(search_query)
        
        print("🤖 Generating Support Response...")
        qa_chain = self.prompt | self.llm | StrOutputParser()
        answer = qa_chain.invoke(
            {"context": context_str, "chat_history": history_str, "question": original_question},
            config=langfuse_config
        )
        
        self.langfuse_client.flush()
        return {"generation": answer, "documents": True if context_str else False}


def initialize_support_rag_pipeline():
    print("🚀 Initializing Voice Support Pipeline...")
    
    # --- THE AUTO-FALLBACK LOGIC ---
    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.path.exists(BM25_SAVE_PATH):
        print("⚠️ Databases not found! Triggering automatic preprocessing...")
        if not os.path.exists(TARGET_PDF_DIR):
            raise FileNotFoundError(f"❌ Cannot build databases. PDF folder missing at: {TARGET_PDF_DIR}")
        
        # Call the function from your preprocessor script
        preprocess_and_index(TARGET_PDF_DIR)
        print("🔄 Preprocessing complete. Resuming engine startup...")

    # --- NEW: Initialize NVIDIA Embeddings ---
    print(f"🧠 Loading NVIDIA Embeddings ({EMBEDDING_MODEL_NAME})...")
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("❌ NVIDIA_API_KEY not found in your .env file!")
        
    embedding_model = NVIDIAEmbeddings(
        model=EMBEDDING_MODEL_NAME
    )

    print("💾 Loading ChromaDB from disk...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR
    )
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    print("💾 Loading BM25 Retriever from disk...")
    with open(BM25_SAVE_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)

    # Initialize Local Reranker
    print(f"⚖️ Initializing Local Reranker: {RERANKER_MODEL_NAME}...")
    cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)

    print("🤖 Initializing Local/Groq LLM...")
    llm = ChatGroq(
        model="llama-3.3-70b-Versatile",
        temperature=0.1
    )

    system_prompt = """You are an expert, helpful AI Customer Support Agent.
    Your role is to guide agents and users based strictly on the provided Zendesk Support Manual.
    
    CRITICAL SUPPORT GUARDRAILS:
    1. STRICT ADHERENCE: Base your answers ONLY on the provided context. Do NOT guess or invent UI navigation steps, pricing, or features that are not explicitly stated in the text.
    2. BE CONCISE FOR VOICE: You are a Voice AI. Keep your answers brief, conversational, and easy to listen to. Do not use complex formatting like markdown tables.
    3. MISSING INFORMATION: If the answer is not in the provided manual, explicitly state: "I don't have that information in my current knowledge base. Would you like me to connect you to a human agent?"

    Previous Conversation History:
    {chat_history}

    Knowledge Base Context:
    {context}

    User Query: {question}
    Support Agent Response:"""
    
    prompt = PromptTemplate.from_template(system_prompt)
    
    print("✅ Advanced Support RAG Pipeline Ready.")
    return ZendeskRAGPipeline(chroma_retriever, bm25_retriever, cross_encoder, prompt, llm)