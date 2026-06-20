import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import warnings

warnings.filterwarnings('ignore')

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

def create_binary_sequences(data_df, seq_length):
    xs, ys = [], []
    feature_cols = [c for c in data_df.columns if c != 'Close']
    feature_data = data_df[feature_cols].values
    close_data = data_df['Close'].values
    
    for i in range(len(data_df) - seq_length):
        x = feature_data[i:(i + seq_length)]
        current_close = close_data[i + seq_length - 1]
        next_close = close_data[i + seq_length]
        
        y = 1.0 if next_close > current_close else 0.0
        
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    # UPDATED FILE PATHS FOR FINBERT MULTIMODAL DATA
    processed_data_path = "data/processed/multimodal_AAPL_ALPHA_2010-01-01_2023-01-01.csv"
    model_path = "models/lstm_classifier.pth"
    raw_data_path = "data/raw/AAPL_ALPHA_2010-01-01_2023-01-01.csv"
    
    print("Loading Multimodal data and AI weights for Backtest...")
    df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
    raw_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    raw_df.dropna(inplace=True)
    
    SEQ_LENGTH = 60
    INPUT_SIZE = len(df.columns) - 1  
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1
    
    X, y_true = create_binary_sequences(df, SEQ_LENGTH)
    
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    test_dates = df.index[train_size + SEQ_LENGTH:]
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = StockClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print("Generating trading signals on unseen data...")
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        probabilities = model(X_test_tensor).cpu().numpy().flatten()
    
    buy_threshold = np.percentile(probabilities, 70)
    sell_threshold = np.percentile(probabilities, 30)
    
    print(f"Dynamic Calibration -> BUY > {buy_threshold:.4f} | SELL < {sell_threshold:.4f}")
    
    signals = np.zeros(len(probabilities))
    current_position = 1 
    
    for i in range(len(probabilities)):
        if probabilities[i] > buy_threshold:    
            current_position = 1
        elif probabilities[i] < sell_threshold:  
            current_position = 0
        
        signals[i] = current_position
    
    print("Running $10,000 Portfolio Simulation (Multimodal Strategy)...") 
    
    bt_df = pd.DataFrame(index=test_dates)
    bt_df['Real_Price'] = raw_df.loc[test_dates, 'Close'].values
    bt_df['Market_Return'] = bt_df['Real_Price'].pct_change()
    bt_df['AI_Signal'] = signals
    
    bt_df['AI_Position'] = bt_df['AI_Signal'].shift(1)
    bt_df.fillna(0, inplace=True)
    bt_df['Trade_Executed'] = bt_df['AI_Position'].diff().abs()
    bt_df['Trade_Executed'] = bt_df['Trade_Executed'].fillna(0)
    
    bt_df['Strategy_Return'] = bt_df['Market_Return'] * bt_df['AI_Position']

    INITIAL_CAPITAL = 10000.0
    
    bt_df['Market_Portfolio'] = INITIAL_CAPITAL * (1 + bt_df['Market_Return']).cumprod()
    bt_df['AI_Portfolio'] = INITIAL_CAPITAL * (1 + bt_df['Strategy_Return']).cumprod()
    
    final_market = bt_df['Market_Portfolio'].iloc[-1]
    final_ai = bt_df['AI_Portfolio'].iloc[-1]
    total_trades = int(bt_df['Trade_Executed'].sum())
    
    ai_total_return = ((final_ai - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    market_total_return = ((final_market - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print("\n" + "="*50)
    print("💰 MULTIMODAL BACKTEST RESULTS")
    print("="*50)
    print(f"Starting Capital:   ${INITIAL_CAPITAL:,.2f}")
    print(f"Total Trades Made:  {total_trades}")
    print(f"Final Market Value: ${final_market:,.2f} ({market_total_return:+.2f}%)")
    print(f"Final AI Value:     ${final_ai:,.2f} ({ai_total_return:+.2f}%)")
    print("="*50)
    
    if final_ai > final_market:
        print("🏆 YOUR MULTIMODAL AI BEAT THE MARKET!")
    else:
        print("📉 The Market beat your AI.")

    print("\nGenerating Equity Curve Chart...")
    plt.figure(figsize=(14, 7))
    
    plt.plot(bt_df.index, bt_df['Market_Portfolio'], label=f"Buy & Hold AAPL (${final_market:,.0f})", color="blue", alpha=0.6, linewidth=2)
    plt.plot(bt_df.index, bt_df['AI_Portfolio'], label=f"Multimodal AI Net (${final_ai:,.0f})", color="green", linewidth=2.5)
    
    plt.title("Multimodal AI Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    plt.gca().yaxis.set_major_formatter(tick)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i in range(1, len(bt_df)):
        if bt_df['AI_Position'].iloc[i] == 0:
            plt.axvspan(bt_df.index[i-1], bt_df.index[i], facecolor='red', alpha=0.1)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/multimodal_backtest.png")
    print("\nSUCCESS! Chart generated at 'results/multimodal_backtest.png'.")