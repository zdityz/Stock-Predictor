# NEXUS

> A neural market intelligence powered by quantitative analysis, deep learning and generative AI.

### 🚀 Live Demo
_______________________

NEXUS is a full-stack financial intelligence platform that combines technical market analysis, machine learning inference, sentiment processing and large language models into a single research workflow.

The platform ingests live market data, extracts momentum and trend signals, evaluates market structure using a neural network and produces institutional-style analyst reports through Google Gemini.

Built with FastAPI, PyTorch, React and Vite.

---

## Overview

Traditional technical analysis tools show indicators.

NEXUS attempts to answer a more useful question:

> **"Given price action, market momentum and current sentiment, what is the most likely market posture right now?"**

To do this, the system combines three independent intelligence layers:

1. **Technical Analysis Engine**

   * RSI
   * MACD
   * Moving Averages
   * Trend Strength Metrics

2. **Neural Prediction Engine**

   * PyTorch LSTM network
   * Multi-feature sequence analysis
   * Rolling historical windows
   * Confidence-weighted outputs

3. **Generative Analyst Engine**

   * Google Gemini
   * Institutional-style market commentary
   * Signal interpretation
   * Risk-aware narrative generation

The result is a quantitative research terminal capable of producing both numerical signals and human-readable analysis.

---

## Architecture

```text
                    ┌─────────────────┐
                    │   React Client  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ FastAPI Backend │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼

 ┌────────────┐      ┌────────────┐      ┌────────────┐
 │ Market Data│      │ TA Engine  │      │ Sentiment  │
 │  Pipeline  │      │ RSI/MACD   │      │  Analysis  │
 └─────┬──────┘      └─────┬──────┘      └─────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼

                  ┌─────────────────┐
                  │  LSTM Network   │
                  │   (PyTorch)     │
                  └────────┬────────┘
                           ▼

                  ┌─────────────────┐
                  │ Google Gemini   │
                  │ Analyst Layer   │
                  └────────┬────────┘
                           ▼

                  Institutional Report
```

---

## Features

### Neural Market Analysis

A custom LSTM model evaluates historical market structure and generates a confidence score representing directional conviction.

### Technical Momentum Engine

Computes multiple quantitative indicators including:

* RSI
* MACD
* SMA
* EMA
* Trend Momentum

### Sentiment Intelligence

Financial headlines are processed through NLP pipelines to estimate market sentiment and institutional positioning.

### AI Analyst Reports

Google Gemini converts quantitative outputs into concise institutional-grade research summaries.

### Global Asset Support

Analyze:

* US Equities
* ETFs
* Cryptocurrency
* NSE Stocks
* BSE Stocks
* Global Indices

Examples:

```text
AAPL
MSFT
SPY
BTC-USD
ETH-USD
RELIANCE.NS
TCS.NS
```

---

## Technology Stack

### Backend

* FastAPI
* Uvicorn
* PyTorch
* Pandas
* NumPy
* yFinance
* TA
* Google GenAI

### Frontend

* React
* Vite
* Tailwind CSS
* Recharts
* Lucide React

### Deployment

* Frontend → Vercel
* Backend → Render

---

## Local Development

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/NEXUS.git
cd NEXUS
```

### Backend Setup

```bash
python3 -m venv venv

source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

Start the API:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Backend:

```text
http://localhost:8000
```

---

### Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

## API

### Analyze Asset

```http
GET /analyze/{ticker}
```

Example:

```http
GET /analyze/AAPL
```

Response:

```json
{
  "ticker": "AAPL",
  "latest_price": 175.43,
  "rsi": 42.15,
  "sentiment_score": 0.345,
  "ai_confidence": 0.684,
  "recent_headlines": [
    "Apple announces new supply chain restructuring..."
  ],
  "analyst_report": "The model indicates accumulation conditions..."
}
```

---

## Why NEXUS Exists

Most retail platforms provide raw indicators.

Most AI platforms provide generic commentary.

NEXUS combines both approaches:

* Quantitative signals
* Neural inference
* Sentiment analysis
* Generative reasoning

into a single research workflow designed for rapid market intelligence.

---

## Disclaimer

NEXUS is an experimental research project.

The outputs generated by the technical analysis engine, neural network, sentiment models, and language models may be inaccurate or incomplete.

Nothing generated by this software should be considered investment advice, trading advice, or financial guidance.

Always perform independent research before making financial decisions.
