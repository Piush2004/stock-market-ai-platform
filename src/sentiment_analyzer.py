import os
import torch
import logging
import pandas as pd
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Union

logger = logging.getLogger(__name__)

class FinBERTSentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None):
        """
        Initializes the FinBERT Sentiment Analyzer.
        
        Parameters:
        -----------
        model_name : str, default 'ProsusAI/finbert'
            Hugging Face model repository ID.
        device : str, optional
            Execution device ('cuda' or 'cpu'). Auto-detects if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Initializing FinBERT Sentiment Analyzer on device: {self.device}")
        
        # Load Hugging Face tokenizer and classification model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Label mapping for ProsusAI/finbert: 0: positive, 1: negative, 2: neutral
        self.labels = ["Positive", "Negative", "Neutral"]

    def analyze_texts(self, texts: List[str], batch_size: int = 16) -> pd.DataFrame:
        """
        Analyzes the sentiment of a list of financial texts (headlines, earnings call summaries, etc.).
        
        Parameters:
        -----------
        texts : List[str]
            List of texts to evaluate.
        batch_size : int, default 16
            Number of texts to process in a single GPU/CPU batch.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing inputs, sentiment categories, probabilities, and a score (-1 to 1).
        """
        if not texts:
            return pd.DataFrame()

        logger.info(f"Analyzing sentiment for {len(texts)} texts in batches of {batch_size}...")
        
        all_sentiment_labels = []
        all_confidences = []
        all_scores = []
        all_pos_probs = []
        all_neg_probs = []
        all_neu_probs = []

        # Process in batches to prevent out-of-memory errors on large lists
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                # Compute probabilities using Softmax
                probabilities = F.softmax(logits, dim=-1).cpu().numpy()
                
            # Parse outputs
            for probs in probabilities:
                pred_class_idx = np.argmax(probs)
                sentiment_label = self.labels[pred_class_idx]
                confidence = probs[pred_class_idx]
                
                # Score between -1 and 1: Prob(Positive) - Prob(Negative)
                score = probs[0] - probs[1]
                
                all_sentiment_labels.append(sentiment_label)
                all_confidences.append(confidence)
                all_scores.append(score)
                all_pos_probs.append(probs[0]) # Positive
                all_neg_probs.append(probs[1]) # Negative
                all_neu_probs.append(probs[2]) # Neutral

        # Construct result DataFrame
        df_results = pd.DataFrame({
            "Text": texts,
            "Sentiment": all_sentiment_labels,
            "Score": all_scores,
            "Confidence": all_confidences,
            "Positive_Prob": all_pos_probs,
            "Negative_Prob": all_neg_probs,
            "Neutral_Prob": all_neu_probs
        })
        
        logger.info("Sentiment analysis complete.")
        return df_results

if __name__ == "__main__":
    # Setup simple console logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Sample financial news headlines and earnings summaries for demonstration
    sample_texts = [
        "Apple Inc. beats quarterly revenue expectations, shares surge 4% in after-hours trading.",
        "Tesla recalls over 100,000 vehicles due to software glitch, stock drops in premarket.",
        "Federal Reserve officials hold interest rates steady as inflation concerns linger.",
        "Nvidia launches new Blackwell chips, projecting massive revenue growth next year.",
        "The recent earnings call summary highlighted that supply chain constraints severely impacted gross margins and led to a downward revision of full-year EPS guidance.",
        "Management's earnings call summary emphasized record-breaking cash flow generation and the successful launch of three new high-margin product lines, increasing dividend payouts."
    ]
    
    # Initialize and execute
    analyzer = FinBERTSentimentAnalyzer()
    df_sentiments = analyzer.analyze_texts(sample_texts)
    
    # Display results
    print("\n--- SENTIMENT ANALYSIS RESULTS ---")
    print(df_sentiments[["Text", "Sentiment", "Score", "Confidence"]].to_string(index=False))
    
    # Save output to models directory
    os.makedirs("models", exist_ok=True)
    output_path = os.path.join("models", "text_sentiments.csv")
    df_sentiments.to_csv(output_path, index=False)
    print(f"\nSaved detailed sentiment scores to: {output_path}")
