import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, start_date, end_date):
    """
    Downloads stock data from Yahoo Finance and saves it to a CSV.
    """
    file_path = f"data/raw/{ticker}_{start_date}_{end_date}.csv"
    
    if os.path.exists(file_path):
        print(f"Loading {ticker} data from local file...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading {ticker} data from Yahoo Finance...")
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        
        # --- THE FIX IS HERE ---
        # If yfinance added a double-header (MultiIndex), drop the second level ('AAPL')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # -----------------------

        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(file_path)
        print(f"Saved to {file_path}")

    return df

if __name__ == "__main__":
    df = fetch_stock_data("AAPL", "2010-01-01", "2023-01-01")
    print("\nFirst 5 rows:")
    print(df.head())