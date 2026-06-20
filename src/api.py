import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
import torch
import joblib
from classifier_model import StockClassifierLSTM
from sentiment_processor import FinBertAnalyzer
from google import genai
from dotenv import load_dotenv
import traceback

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("models", exist_ok=True)

try:
    api_key_env = os.getenv("GEMINI_API_KEY")
    if not api_key_env:
        print("Warning: GEMINI_API_KEY environment variable not found in .env file.")
        
    client = genai.Client(api_key=api_key_env)
    analyzer = FinBertAnalyzer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model_path = "models/lstm_classifier.pth"
    scaler_path = "models/multimodal_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = StockClassifierLSTM(input_size=10, hidden_size=50, num_layers=2, output_size=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        scaler = joblib.load(scaler_path)
        print("AI Core Operational: Environment variables configured correctly.")
    else:
        print("Warning: Model weights or scaler files not found in models/ directory.")
except Exception as e:
    print(f"Core Initialization Failure: {e}")

def generate_analyst_commentary(ticker, confidence, sentiment, rsi):
    try:
        prompt = f"""
        You are a professional financial analyst. Analyze these metrics for {ticker}:
        - AI Predictive Confidence: {confidence:.2f} (Scale 0-1, where 1 is strong buy)
        - News Sentiment Score: {sentiment:.2f} (Scale -1 to 1)
        - RSI: {rsi:.2f} (Above 70 is overbought, below 30 is oversold)
        
        Provide a concise, professional 3-sentence summary of whether this stock is currently an accumulation or a defensive hold. Use professional, analytical tone.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as gemini_err:
        return f"Analyst report temporarily unavailable: {str(gemini_err)}"

@app.get("/")
def home():
    return {"status": "online", "message": "Stock Predictor Live API running cleanly."}

@app.get("/analyze/{ticker}")
def analyze_stock(ticker: str):
    try:
        target_data = yf.download(ticker, period="200d", interval="1d", progress=False)
        spy_data = yf.download("SPY", period="200d", interval="1d", progress=False)
        vix_data = yf.download("^VIX", period="200d", interval="1d", progress=False)

        if target_data.empty:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found.")

        if isinstance(target_data.columns, pd.MultiIndex):
            target_data.columns = target_data.columns.get_level_values(0)
            spy_data.columns = spy_data.columns.get_level_values(0)
            vix_data.columns = vix_data.columns.get_level_values(0)

        target_data.index = pd.to_datetime(target_data.index).tz_localize(None).normalize()
        spy_data.index = pd.to_datetime(spy_data.index).tz_localize(None).normalize()
        vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None).normalize()

        df = pd.DataFrame(index=target_data.index)
        df['Close'] = target_data['Close']
        df['Volume'] = target_data['Volume']
        df['SPY_Close'] = spy_data['Close']
        df['VIX_Close'] = vix_data['Close']
        df.ffill(inplace=True)
        df.bfill(inplace=True)

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

        headlines = []
        try:
            news_ticker = yf.Ticker(ticker)
            raw_news = news_ticker.news
            if raw_news and isinstance(raw_news, list):
                for item in raw_news[:6]:
                    title = item.get("title") or item.get("content", {}).get("title") or item.get("summary")
                    if title and isinstance(title, str):
                        headlines.append(title.strip())
        except Exception as news_err:
            print(f"Warning: News parsing anomaly for {ticker}: {news_err}")

        if headlines:
            sentiment_score = analyzer.get_sentiment_score(headlines)
        else:
            headlines = ["No recent context available via market provider feeds."]
            sentiment_score = 0.0000

        df['Sentiment_Score'] = sentiment_score
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        latest_data = df.iloc[-60:].copy()

        feature_columns = [
            'Return_1d', 'Return_5d', 'Dist_SMA20', 'Dist_SMA50',
            'RSI_14', 'MACD_Hist', 'Volume_Change', 'SPY_Return',
            'VIX_Change', 'Sentiment_Score'
        ]

        scaled_features = scaler.transform(latest_data[feature_columns])
        x_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            probability = model(x_tensor).cpu().numpy().flatten()[0]

        latest_close = float(df['Close'].iloc[-1])
        latest_rsi = float(df['RSI_14'].iloc[-1])

        analyst_report = generate_analyst_commentary(ticker.upper(), probability, sentiment_score, latest_rsi)

        return {
            "ticker": ticker.upper(),
            "latest_price": latest_close,
            "rsi": latest_rsi,
            "sentiment_score": float(sentiment_score),
            "recent_headlines": headlines,
            "ai_confidence": float(probability),
            "analyst_report": analyst_report
        }
    except Exception as e:
        print("\n=== CRITICAL BACKEND ERROR ===")
        traceback.print_exc()
        print("==============================\n")
        raise HTTPException(status_code=500, detail=str(e))