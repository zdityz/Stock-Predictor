import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. RECREATE THE MODEL ARCHITECTURE ---
class StockPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length, target_col_index):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_col_index] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    processed_data_path = "data/processed/scaled_AAPL_2010-01-01_2023-01-01.csv"
    model_path = "models/lstm_regressor.pth"
    raw_data_path = "data/raw/AAPL_2010-01-01_2023-01-01.csv"
    
    print("Loading data and AI weights...")
    df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
    columns = list(df.columns)
    log_return_idx = columns.index('Log_Return')
    data_array = df.values
    
    # Recreate the scaler JUST for Log_Return so we can un-scale it back to real percentages
    raw_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    raw_df['Log_Return'] = np.log(raw_df['Close'] / raw_df['Close'].shift(1))
    raw_df.dropna(inplace=True) 
    
    return_scaler = MinMaxScaler(feature_range=(0, 1))
    return_scaler.fit(raw_df[['Log_Return']]) 

    # Hyperparameters
    SEQ_LENGTH = 60
    INPUT_SIZE = len(columns)
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1
    
    X, y_true = create_sequences(data_array, SEQ_LENGTH, target_col_index=log_return_idx)
    
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    y_test_true = y_true[train_size:]
    test_dates = df.index[train_size + SEQ_LENGTH:]
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = StockPredictorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() 
    
    print("Making predictions on unseen data...")
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy()
    
    print("Converting predictions back to real percentages...")
    predictions_real = return_scaler.inverse_transform(predictions)
    y_test_true_real = return_scaler.inverse_transform(y_test_true.reshape(-1, 1))
    
    # To make the chart readable, let's just plot the last 100 days of the test set
    plot_days = 100
    
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates[-plot_days:], y_test_true_real[-plot_days:], label="Actual Daily Return", color="blue", linewidth=2)
    plt.plot(test_dates[-plot_days:], predictions_real[-plot_days:], label="AI Predicted Return", color="red", linestyle="dashed", linewidth=1.5)
    
    # Add a horizontal line at 0 (the difference between a winning day and a losing day)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.title(f"AI Model vs Reality: Predicting Daily Returns (Last {plot_days} Days)")
    plt.xlabel("Date")
    plt.ylabel("Log Return (0.01 = ~1%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/return_prediction_chart.png")
    print("\nSUCCESS! Open 'results/return_prediction_chart.png'.")