import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# --- 1. DATA PREPARATION ---
def create_sequences(data, seq_length, target_col_index):
    """
    Creates sequences of length `seq_length` to predict the next day's target.
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        # Grab the target column from the next day (e.g., Log_Return)
        y = data[i + seq_length, target_col_index] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- 2. THE NEURAL NETWORK (LSTM) ---
class StockPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Linear layer outputting a single continuous value
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 3. THE TRAINING ENGINE ---
if __name__ == "__main__":
    data_path = "data/processed/scaled_AAPL_2010-01-01_2023-01-01.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Did you run features.py?")
        exit()

    print("Loading processed data...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Identify the index for 'Log_Return'
    # Assuming the columns match exactly what we built in features.py
    columns = list(df.columns)
    try:
        log_return_idx = columns.index('Log_Return')
    except ValueError:
        print("Error: 'Log_Return' column not found. Check features.py output.")
        exit()

    data_array = df.values 
    
    # --- HYPERPARAMETERS ---
    SEQ_LENGTH = 60
    INPUT_SIZE = len(columns) 
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1         
    LEARNING_RATE = 0.001   
    EPOCHS = 40             
    BATCH_SIZE = 32         
    
    print(f"Generating sequences targeting index {log_return_idx} ('Log_Return')...")
    X, y = create_sequences(data_array, SEQ_LENGTH, target_col_index=log_return_idx)
    
    # Split
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = StockPredictorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    
    # Using Mean Squared Error because we are predicting a continuous value
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Training Phase (Regression on Log Return)...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    os.makedirs("models", exist_ok=True)
    # Save with a specific name so we don't overwrite a classifier later
    torch.save(model.state_dict(), "models/lstm_regressor.pth")
    print("\nTraining complete! Model saved to models/lstm_regressor.pth")