import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os

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

def create_binary_sequences(data, seq_length, close_col_index):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        current_close = data[i + seq_length - 1, close_col_index]
        next_close = data[i + seq_length, close_col_index]
        y = 1.0 if next_close > current_close else 0.0
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    processed_data_path = "data/processed/scaled_AAPL_2010-01-01_2023-01-01.csv"
    model_path = "models/lstm_classifier.pth"
    raw_data_path = "data/raw/AAPL_2010-01-01_2023-01-01.csv"
    
    print("Loading data and AI weights...")
    df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
    raw_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    raw_df.dropna(inplace=True)
    
    close_idx = list(df.columns).index('Close')
    data_array = df.values
    
    SEQ_LENGTH = 60
    INPUT_SIZE = len(df.columns)
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1
    
    X, y_true = create_binary_sequences(data_array, SEQ_LENGTH, close_col_index=close_idx)
    
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    y_test_true = y_true[train_size:]
    
    test_dates = df.index[train_size + SEQ_LENGTH:]
    real_prices = raw_df.loc[test_dates, 'Close'].values
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = StockClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print("Running AI on unseen market data...")
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        probabilities = model(X_test_tensor).cpu().numpy().flatten()

    predictions = (probabilities > 0.5).astype(int)

    accuracy = accuracy_score(y_test_true, predictions) * 100
    print(f"\n--- MODEL RESULTS ---")
    print(f"True Test Accuracy: {accuracy:.2f}%")
    print(f"(Remember: In stock markets, anything consistently over 53% is an edge!)")
    
    plot_days = 150
    plot_dates = test_dates[-plot_days:]
    plot_prices = real_prices[-plot_days:]
    plot_preds = predictions[-plot_days:]
    
    plt.figure(figsize=(14, 7))
    plt.plot(plot_dates, plot_prices, label="Actual AAPL Price", color="black", linewidth=1.5, alpha=0.7)
    
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    
    for i in range(len(plot_preds)):
        if plot_preds[i] == 1:
            buy_dates.append(plot_dates[i])
            buy_prices.append(plot_prices[i])
        else:
            sell_dates.append(plot_dates[i])
            sell_prices.append(plot_prices[i])
            
    plt.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label="AI Predicts UP (Buy)")
    plt.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label="AI Predicts DOWN (Sell)")
    
    plt.title(f"AI Classifier Signals: Last {plot_days} Trading Days")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/classifier_signals_chart.png")
    print("\nSUCCESS! Open 'results/classifier_signals_chart.png' to see your AI's trading signals.")