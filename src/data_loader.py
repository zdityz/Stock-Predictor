import yfinance as yf
import pandas as pd
import os

def fetch_advanced_data(ticker, start_date, end_date):
    print(f"Building the Alpha Dataset for {ticker}...")
    
    target_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if isinstance(target_data.columns, pd.MultiIndex):
        target_data.columns = target_data.columns.get_level_values(0)
        
    spy_data = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True, progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
        
    vix_data = yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    df = pd.DataFrame(index=target_data.index)
    df['Close'] = target_data['Close']
    df['Volume'] = target_data['Volume']
    df['SPY_Close'] = spy_data['Close']
    df['VIX_Close'] = vix_data['Close']
    
    df.dropna(inplace=True)
    
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{ticker}_ALPHA_{start_date}_{end_date}.csv"
    df.to_csv(file_path)
    print(f"Saved Alpha dataset to {file_path}")
    return df

if __name__ == "__main__":
    fetch_advanced_data("AAPL", "2010-01-01", "2023-01-01")