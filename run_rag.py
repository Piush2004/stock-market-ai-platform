import os
import sys
import argparse
import logging

from src.rag_pipeline import FinancialRAGPipeline

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Financial Q&A RAG CLI (Ollama, ChromaDB, Sentence-Transformers)")
    parser.add_argument("--persist-dir", type=str, default="./data/chroma_db", help="ChromaDB persistence directory")
    parser.add_argument("--model", type=str, default="llama3", help="Global default Ollama model (default: llama3)")
    parser.add_argument("--url", type=str, default="http://localhost:11434", help="Global default Ollama server URL")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a financial report (PDF), SEC filing (HTML), or transcript (TXT)")
    ingest_parser.add_argument("--file", type=str, required=True, help="Path to the PDF, HTML, or TXT file to ingest")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline for a question")
    query_parser.add_argument("question", type=str, help="The question to ask the RAG pipeline")
    query_parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    query_parser.add_argument("--model", type=str, dest="sub_model", help="Ollama LLM model to query (overrides global)")
    query_parser.add_argument("--url", type=str, dest="sub_url", help="Ollama server URL (overrides global)")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive financial Q&A session")
    chat_parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    chat_parser.add_argument("--model", type=str, dest="sub_model", help="Ollama LLM model to query (overrides global)")
    chat_parser.add_argument("--url", type=str, dest="sub_url", help="Ollama server URL (overrides global)")
    
    args = parser.parse_args()
    
    # Resolve overrides (use subcommand model/url if specified, else global defaults)
    selected_model = getattr(args, "sub_model", None) or args.model
    selected_url = getattr(args, "sub_url", None) or args.url
    
    # Initialize RAG Pipeline
    pipeline = FinancialRAGPipeline(
        persist_dir=args.persist_dir,
        ollama_url=selected_url,
        ollama_model=selected_model
    )
    
    # Verify Ollama server
    if args.command in ["query", "chat"]:
        logger = logging.getLogger("RAG-CLI")
        if not pipeline.verify_ollama():
            logger.warning(
                f"\n[WARNING] Could not connect to local Ollama server at {selected_url}.\n"
                f"Please ensure the Ollama application is running on your computer.\n"
                f"Commands: Run `ollama serve` and make sure you have pulled your model: `ollama pull {selected_model}`.\n"
            )

    if args.command == "ingest":
        try:
            chunks = pipeline.ingest_document(args.file)
            print(f"\n[Success] Ingested '{args.file}'. Created and embedded {chunks} chunks inside ChromaDB.")
        except Exception as e:
            print(f"\n[Error] Failed to ingest file: {str(e)}")
            sys.exit(1)
            
    elif args.command == "query":
        print(f"\nAsking local LLM ({selected_model}) with context...")
        result = pipeline.query(args.question, k=args.k)
        
        print("\n=== ANSWER ===")
        print(result["answer"])
        print("\n=== SOURCES ===")
        seen_sources = set()
        for idx, src in enumerate(result["sources"]):
            src_str = f"- {src['file']}"
            if 'page' in src:
                src_str += f" (Page {src['page']})"
            src_str += f" [Similarity Score: {src['score']:.4f}]"
            
            # Avoid printing exact duplicate pages
            if src_str not in seen_sources:
                print(src_str)
                seen_sources.add(src_str)
                
    elif args.command == "chat":
        print(f"\nStarting interactive RAG session using Ollama model: '{selected_model}'")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                question = input("\nAsk a question: ").strip()
                if not question:
                    continue
                if question.lower() in ["exit", "quit"]:
                    print("Exiting chat session.")
                    break
                    
                print("Retrieving context and generating answer...")
                result = pipeline.query(question, k=args.k)
                
                print("\n=== ANSWER ===")
                print(result["answer"])
                print("\n=== SOURCES ===")
                seen_sources = set()
                for src in result["sources"]:
                    src_str = f"- {src['file']}"
                    if 'page' in src:
                        src_str += f" (Page {src['page']})"
                    if src_str not in seen_sources:
                        print(src_str)
                        seen_sources.add(src_str)
            except KeyboardInterrupt:
                print("\nExiting chat session.")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
