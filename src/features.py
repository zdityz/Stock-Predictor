import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.preprocessing import MinMaxScaler
import os

def engineer_features(raw_csv_path):
    print(f"Loading raw data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
    
    print("Calculating Technical Indicators...")

    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    print("Calculating Log Returns...")
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    print("Scaling data to range [0, 1]...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_columns = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'Log_Return']
    scaled_data = scaler.fit_transform(df[feature_columns])
    df_scaled = pd.DataFrame(scaled_data, columns=feature_columns, index=df.index)
    
    os.makedirs("data/processed", exist_ok=True)
    
    filename = os.path.basename(raw_csv_path)
    save_path = f"data/processed/scaled_{filename}"
    
    df_scaled.to_csv(save_path)
    print(f"Feature engineering complete. Saved to: {save_path}")
    
    return df_scaled, scaler

if __name__ == "__main__":
    raw_file = "data/raw/AAPL_2010-01-01_2023-01-01.csv" 
    
    if os.path.exists(raw_file):
        processed_df, saved_scaler = engineer_features(raw_file)
        
        print("\nProcessed Data (First 5 rows):")
        print(processed_df.head())
    else:
        print(f"Error: Could not find {raw_file}. Did you run data_loader.py first?")