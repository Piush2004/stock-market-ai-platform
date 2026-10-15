import os
import requests
import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class FinancialRAGPipeline:
    def __init__(
        self, 
        persist_dir: str = "./data/chroma_db", 
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3"
    ):
        """
        Initializes the RAG Pipeline.
        """
        self.persist_dir = persist_dir
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        
        logger.info(f"Initializing embeddings model: {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'} # standard CPU execution for embeddings
        )
        
        logger.info(f"Connecting to ChromaDB at {persist_dir}...")
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def verify_ollama(self) -> bool:
        """
        Checks if the local Ollama instance is running.
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                logger.info(f"Ollama is running. Available models: {models}")
                return True
        except Exception:
            pass
        return False

    def ingest_document(self, file_path: str) -> int:
        """
        Loads and chunks a document (PDF, HTML, or Text), embeds the chunks, and saves to ChromaDB.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"Ingesting document: {file_path} (Type: {file_ext})...")
        
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext in [".txt", ".log", ".csv"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_ext in [".html", ".htm"]:
            loader = BSHTMLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Only PDF, HTML, and text files are supported.")
            
        # Load documents and split
        raw_docs = loader.load()
        chunks = self.text_splitter.split_documents(raw_docs)
        
        # Add metadata like source filename
        for chunk in chunks:
            chunk.metadata["source_file"] = os.path.basename(file_path)
            
        logger.info(f"Created {len(chunks)} chunks from {os.path.basename(file_path)}.")
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        logger.info(f"Successfully embedded and stored chunks in ChromaDB.")
        return len(chunks)

    def retrieve_context(self, question: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant chunks from ChromaDB for a query.
        """
        results = self.vector_store.similarity_search_with_relevance_scores(question, k=k)
        
        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        return retrieved_docs

    def call_ollama(self, prompt: str) -> str:
        """
        Queries the local Ollama LLM endpoint directly.
        """
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2 # Lower temperature for more factual financial Q&A
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"Error: Ollama returned status code {response.status_code}. Response: {response.text}"
        except requests.exceptions.ConnectionError:
            return (
                f"Connection Error: Could not connect to local Ollama server at {self.ollama_url}.\n"
                f"Please verify Ollama is running (`ollama serve`) and the model '{self.ollama_model}' is pulled."
            )
        except Exception as e:
            return f"Error calling local Ollama LLM: {str(e)}"

    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline:
        1. Retrieves relevant context.
        2. Constructs the prompt.
        3. Invokes local Ollama LLM.
        """
        logger.info(f"Retrieving context for query: '{question}'...")
        contexts = self.retrieve_context(question, k=k)
        
        if not contexts:
            return {
                "answer": "No relevant document context found in the database. Please ingest some documents first.",
                "sources": []
            }
            
        # Construct context string
        context_str = ""
        for idx, ctx in enumerate(contexts):
            src_info = ctx['metadata'].get('source_file', 'Unknown Source')
            page_info = f", Page {ctx['metadata'].get('page', 0) + 1}" if 'page' in ctx['metadata'] else ""
            context_str += f"[Source {idx+1}: {src_info}{page_info}]\n{ctx['content']}\n\n"
            
        # Construct RAG Prompt
        prompt = (
            "You are an expert financial analyst. Answer the user's question accurately and objectively using "
            "only the provided contexts. If the context does not contain the answer, state that you do not know "
            "based on the provided context. Do not invent facts.\n\n"
            f"=== CONTEXTS ===\n{context_str}\n"
            f"=== QUESTION ===\n{question}\n\n"
            "=== ANSWER ===\n"
        )
        
        logger.info("Invoking Ollama LLM...")
        answer = self.call_ollama(prompt)
        
        # Format sources
        sources = []
        for ctx in contexts:
            src = {
                "file": ctx['metadata'].get('source_file', 'Unknown'),
                "score": ctx['score']
            }
            if 'page' in ctx['metadata']:
                src["page"] = ctx['metadata']['page'] + 1
            sources.append(src)
            
        return {
            "answer": answer,
            "sources": sources
        }
