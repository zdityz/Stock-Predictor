import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.preprocessing import StandardScaler
from sentiment_processor import FinBertAnalyzer
import os
import joblib

def engineer_multimodal_features(raw_csv_path):
    print(f"Loading raw data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
    
    df['Return_1d'] = df['Close'].pct_change()
    df['Return_5d'] = df['Close'].pct_change(5)
    
    sma20 = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    sma50 = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['Dist_SMA20'] = (df['Close'] - sma20) / sma20
    df['Dist_SMA50'] = (df['Close'] - sma50) / sma50
    
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD_Hist'] = macd.macd_diff()
    
    df['Volume_Change'] = df['Volume'].pct_change()
    df['SPY_Return'] = df['SPY_Close'].pct_change()
    df['VIX_Change'] = df['VIX_Close'].diff()
    
    print("Integrating FinBERT Sentiment Layer...")
    analyzer = FinBertAnalyzer()
    df_sentiment = analyzer.generate_synthetic_historical_sentiment(df.index)
    df = df.join(df_sentiment)
    
    df.dropna(inplace=True)
    
    feature_columns = [
        'Return_1d', 'Return_5d', 'Dist_SMA20', 'Dist_SMA50', 
        'RSI_14', 'MACD_Hist', 'Volume_Change', 'SPY_Return', 
        'VIX_Change', 'Sentiment_Score'
    ]
    
    train_size = int(len(df) * 0.8)
    
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_size][feature_columns])
    
    scaled_features = scaler.transform(df[feature_columns])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_columns, index=df.index)
    
    df_scaled['Close'] = df['Close'].values
    
    os.makedirs("data/processed", exist_ok=True)
    
    filename = os.path.basename(raw_csv_path)
    save_path = f"data/processed/multimodal_{filename}"
    
    df_scaled.to_csv(save_path)
    joblib.dump(scaler, "models/multimodal_scaler.pkl")
    
    print(f"Multimodal features complete. Saved to: {save_path}")
    return df_scaled

if __name__ == "__main__":
    raw_file = "data/raw/AAPL_ALPHA_2010-01-01_2023-01-01.csv"
    if os.path.exists(raw_file):
        engineer_multimodal_features(raw_file)
    else:
        print("Data file not found.")