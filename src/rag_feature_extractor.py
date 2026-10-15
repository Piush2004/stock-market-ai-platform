import os
import json
import logging
import pandas as pd
from typing import Dict, Any

from src.rag_pipeline import FinancialRAGPipeline

logger = logging.getLogger(__name__)

class RAGFeatureExtractor:
    def __init__(self, persist_dir: str = "./data/chroma_db", ollama_model: str = "llama3"):
        self.pipeline = FinancialRAGPipeline(
            persist_dir=persist_dir,
            ollama_model=ollama_model
        )
        self.output_file = "data/processed/rag_features.csv"
        
        # Define the aspects to extract and the specific question to ask
        self.aspects = {
            "Revenue_Growth": "What is the recent revenue growth reported?",
            "Profit_Growth": "What is the recent profit growth reported?",
            "Guidance": "What is the future guidance or outlook provided by management?",
            "Risk_Factors": "What are the key risk factors mentioned?",
            "Debt_Info": "What is the current debt information or leverage situation?"
        }
        
    def _extract_score(self, question: str, ticker: str) -> int:
        """
        Retrieves context for a ticker and asks the LLM to rate the aspect 
        as Positive (1), Neutral (0), or Negative (-1).
        """
        # Since the vector store contains all documents, we append the ticker to guide retrieval.
        targeted_question = f"Regarding {ticker}: {question}"
        
        logger.info(f"Extracting '{question}' for {ticker}...")
        
        contexts = self.pipeline.retrieve_context(targeted_question, k=4)
        if not contexts:
            logger.warning(f"No context found for {ticker}. Returning Neutral (0).")
            return 0
            
        context_str = ""
        for idx, ctx in enumerate(contexts):
            context_str += f"[Source {idx+1}]\n{ctx['content']}\n\n"
            
        # Strict prompt to get only an integer
        prompt = (
            "You are a quantitative financial analyst. Based ONLY on the provided context, "
            "evaluate the sentiment or implication of the following aspect for the company.\n\n"
            "Respond with EXACTLY ONE INTEGER and nothing else:\n"
            " 1  if the context implies a positive/favorable situation (e.g. growing revenue, optimistic guidance, low debt risk).\n"
            " 0  if the context is neutral, mixed, or the information is not present.\n"
            "-1  if the context implies a negative/unfavorable situation (e.g. declining profit, pessimistic guidance, high debt risk).\n\n"
            f"=== CONTEXTS ===\n{context_str}\n"
            f"=== ASPECT TO EVALUATE ===\n{question}\n\n"
            "=== INTEGER RESPONSE ONLY (-1, 0, or 1) ===\n"
        )
        
        response = self.pipeline.call_ollama(prompt)
        
        # Parse the response safely
        try:
            # Strip any punctuation or whitespace
            cleaned_response = "".join(c for c in response if c in "-0123456789").strip()
            score = int(cleaned_response)
            # Bound the score
            score = max(-1, min(1, score))
            return score
        except ValueError:
            logger.error(f"Failed to parse LLM response for {ticker} as integer. Response was: '{response}'. Defaulting to 0.")
            return 0

    def extract_features_for_tickers(self, tickers: list):
        """
        Runs the extraction for all tickers and saves to a CSV.
        """
        results = []
        
        for ticker in tickers:
            logger.info(f"--- Extracting RAG features for {ticker} ---")
            ticker_features = {"Ticker": ticker}
            
            for aspect_name, question in self.aspects.items():
                score = self._extract_score(question, ticker)
                ticker_features[f"RAG_{aspect_name}"] = score
                
            results.append(ticker_features)
            
        df = pd.DataFrame(results)
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Successfully saved RAG features to {self.output_file}")
        
        return df

    def integrate_with_timeseries(self, features_dir: str = "data/processed"):
        """
        Merges the extracted static RAG features into the daily time-series CSVs.
        """
        if not os.path.exists(self.output_file):
            logger.error("RAG features CSV not found. Run extract_features_for_tickers first.")
            return
            
        rag_df = pd.read_csv(self.output_file)
        
        for index, row in rag_df.iterrows():
            ticker = row["Ticker"]
            ts_path = os.path.join(features_dir, f"{ticker}_features.csv")
            
            if os.path.exists(ts_path):
                ts_df = pd.read_csv(ts_path)
                
                # Assign RAG features as static columns to all historical rows
                for col in rag_df.columns:
                    if col != "Ticker":
                        ts_df[col] = row[col]
                        
                ts_df.to_csv(ts_path, index=False)
                logger.info(f"Integrated RAG features into {ts_path}")
            else:
                logger.warning(f"Time-series data for {ticker} not found at {ts_path}.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # You must run this on your HOST machine where Ollama is actively running.
    extractor = RAGFeatureExtractor(ollama_model="llama3") # Change to your pulled model if needed (e.g. qwen2.5:8b)
    
    TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS"]
    extractor.extract_features_for_tickers(TICKERS)
    extractor.integrate_with_timeseries()
