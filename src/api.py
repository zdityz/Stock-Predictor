from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="AI Stock Oracle API", description="Deep Learning Trading Signals")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/predict/{ticker}")
def get_prediction(ticker: str):
    try:
        df = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        feature_columns = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'Log_Return']
        scaled_data = scaler.fit_transform(df[feature_columns])

        SEQ_LENGTH = 60
        if len(scaled_data) < SEQ_LENGTH:
             raise HTTPException(status_code=400, detail="Not enough data to form a 60-day sequence.")

        recent_sequence = scaled_data[-SEQ_LENGTH:]
        x_tensor = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model_path = "models/lstm_classifier.pth"
        
        if not os.path.exists(model_path):
             raise HTTPException(status_code=500, detail="AI Model not found. Train the model first.")

        model = StockClassifierLSTM(input_size=len(feature_columns), hidden_size=50, num_layers=2, output_size=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            x_tensor = x_tensor.to(device)
            probability = model(x_tensor).item()

        if probability > 0.60:
            signal = "BUY"
        elif probability < 0.40:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "ticker": ticker.upper(),
            "latest_price": round(float(df['Close'].iloc[-1]), 2),
            "prediction_probability": round(probability * 100, 2),
            "signal": signal
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))