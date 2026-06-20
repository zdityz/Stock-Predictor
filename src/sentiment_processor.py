import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import os

class FinBertAnalyzer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        self.model.eval()

    def get_sentiment_score(self, text_list):
        if not text_list:
            return 0.0
        
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        scores = outputs.logits.cpu().numpy()
        scores = softmax(scores, axis=1)
        
        pos = scores[:, 0]
        neg = scores[:, 1]
        neu = scores[:, 2]
        
        net_sentiment = pos - neg
        return float(np.mean(net_sentiment))

    def generate_synthetic_historical_sentiment(self, date_index):
        np.random.seed(42)
        base_sentiment = np.random.normal(0.02, 0.15, size=len(date_index))
        
        df_sentiment = pd.DataFrame(index=date_index)
        df_sentiment['Sentiment_Score'] = np.clip(base_sentiment, -1.0, 1.0)
        
        return df_sentiment

if __name__ == "__main__":
    analyzer = FinBertAnalyzer()
    
    sample_headlines = [
        "Apple reports record-breaking Q4 earnings, smashing analyst expectations.",
        "Supply chain disruptions in Asia could delay the next iPhone shipment.",
        "AAPL shares trade flat following neutral federal regulatory update."
    ]
    
    for headline in sample_headlines:
        score = analyzer.get_sentiment_score([headline])
        print(f"Headline: {headline}")
        print(f"Net Sentiment Score: {score:.4f}\n")