import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

# Suppress annoying terminal warnings for a clean interface
warnings.filterwarnings('ignore')

# --- 1. RECREATE THE CLASSIFIER ARCHITECTURE ---
class StockClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockClassifierLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

def get_live_prediction(ticker="AAPL"):
    print(f"📡 Fetching live market data for {ticker}...")
    # We download 10 years to ensure our MinMaxScaler matches the training scale perfectly
    df = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
    
    # Fix the yfinance MultiIndex issue if it appears
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print("⚙️ Calculating technical indicators...")
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df.dropna(inplace=True)

    print("⚖️ Normalizing data for the AI...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_columns = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'Log_Return']
    scaled_data = scaler.fit_transform(df[feature_columns])

    # Grab EXACTLY the last 60 days of the market
    SEQ_LENGTH = 60
    recent_sequence = scaled_data[-SEQ_LENGTH:]
    
    # Convert to PyTorch Tensor format: (Batch Size, Sequence Length, Features) -> (1, 60, 8)
    x_tensor = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0)

    print("🧠 Waking up the Neural Network...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model_path = "models/lstm_classifier.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: Cannot find {model_path}. You must train the model first.")
        return

    model = StockClassifierLSTM(input_size=len(feature_columns), hidden_size=50, num_layers=2, output_size=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # Tell it to predict, not learn

    # --- THE PREDICTION ---
    with torch.no_grad():
        x_tensor = x_tensor.to(device)
        probability = model(x_tensor).item()

    # --- THE VERDICT ---
    print("\n" + "="*50)
    print(f"🔮 THE DAILY ORACLE: {ticker}")
    print("="*50)
    
    latest_close = df['Close'].iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    print(f"Last Market Close ({latest_date}): ${latest_close:.2f}")
    print(f"AI 'UP' Probability for Next Session: {probability * 100:.2f}%\n")

    if probability > 0.60:
        print("🟢 SIGNAL: HIGH CONVICTION BUY")
        print("The AI sees a strong upward trend forming. Prepare to go long.")
    elif probability < 0.40:
        print("🔴 SIGNAL: HIGH CONVICTION SELL")
        print("The AI detects heavy downward momentum. Move to cash or short.")
    else:
        print("🟡 SIGNAL: HOLD / NO CONVICTION")
        print("The market is noisy right now. The AI lacks conviction. Wait.")
    print("="*50 + "\n")

if __name__ == "__main__":
    # You can change this ticker to test it on other stocks, 
    # but remember the AI was specifically trained on AAPL's behavior!
    get_live_prediction("AAPL")