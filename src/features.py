import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.preprocessing import MinMaxScaler
import os

def engineer_features(raw_csv_path):
    """
    Reads raw stock data, calculates technical indicators, 
    normalizes the data, and saves it for the model.
    """
    print(f"Loading raw data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
    
    # --- 1. TECHNICAL INDICATORS ---
    print("Calculating Technical Indicators...")
    
    # Simple Moving Averages (Trend)
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    # Relative Strength Index (Momentum)
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD (Trend & Momentum)
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # --- 2. STATIONARITY (Log Returns) ---
    # We predict the percentage change, not the raw price.
    print("Calculating Log Returns...")
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # --- 3. CLEANUP ---
    # Moving averages and shifted data create 'NaN' (Not a Number) empty rows at the start.
    # A neural network will crash if it sees a NaN. We must drop them.
    df.dropna(inplace=True)
    
    # --- 4. NORMALIZATION (Scaling) ---
    print("Scaling data to range [0, 1]...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # We define the columns we want our model to learn from
    feature_columns = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'Log_Return']
    
    # Scale only the features we need
    scaled_data = scaler.fit_transform(df[feature_columns])
    
    # Convert back to a DataFrame so it's easy to read and save
    df_scaled = pd.DataFrame(scaled_data, columns=feature_columns, index=df.index)
    
    # --- 5. SAVE ---
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    # Extract the filename from the path to save it cleanly
    filename = os.path.basename(raw_csv_path)
    save_path = f"data/processed/scaled_{filename}"
    
    df_scaled.to_csv(save_path)
    print(f"Feature engineering complete. Saved to: {save_path}")
    
    return df_scaled, scaler

if __name__ == "__main__":
    # Point it to the file we downloaded in Step 1
    # Note: Ensure this exact filename matches what is in your data/raw folder!
    raw_file = "data/raw/AAPL_2010-01-01_2023-01-01.csv" 
    
    if os.path.exists(raw_file):
        processed_df, saved_scaler = engineer_features(raw_file)
        
        print("\nProcessed Data (First 5 rows):")
        # Notice how all numbers are now decimals between 0 and 1
        print(processed_df.head())
    else:
        print(f"Error: Could not find {raw_file}. Did you run data_loader.py first?")