import requests
import time

def test_live_engine(ticker):
    url = f"http://localhost:8000/analyze/{ticker}"
    print(f"Fetching live data and AI analysis for {ticker}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        print("\n" + "="*50)
        print(f" LIVE AI ANALYSIS: {data['ticker']}")
        print("="*50)
        print(f"Latest Price:      ${data['latest_price']:.2f}")
        print(f"RSI (14-day):      {data['rsi']:.2f}")
        print(f"Sentiment Score:   {data['sentiment_score']:.4f}")
        print(f"AI Confidence:     {data['ai_confidence']:.4f} / 1.0")
        
        print("\n🤖 AI ANALYST COMMENTARY:")
        print(data['analyst_report'])
        
        print("\nRecent Headlines:")
        for headline in data['recent_headlines']:
            print(f" - {headline}")
        print("="*50 + "\n")
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the API: {e}")

if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "NVDA"]
    for t in tickers:
        test_live_engine(t)
        print("Waiting 2 seconds to respect API rate limits...\n")
        time.sleep(2)