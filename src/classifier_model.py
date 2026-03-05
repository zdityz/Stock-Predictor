import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# --- 1. DATA PREPARATION (The Binary Shift) ---
def create_binary_sequences(data, seq_length, close_col_index):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        
        # Today's Close Price
        current_close = data[i + seq_length - 1, close_col_index]
        # Tomorrow's Close Price
        next_close = data[i + seq_length, close_col_index]
        
        # THE MAGIC LOGIC: 1 if it goes up, 0 if it goes down
        y = 1.0 if next_close > current_close else 0.0
        
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- 2. THE NEURAL NETWORK (Classifier) ---
class StockClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockClassifierLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # Add a Sigmoid activation to squash the output between 0% and 100% probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out) # Squashes the final number
        return out

if __name__ == "__main__":
    data_path = "data/processed/scaled_AAPL_2010-01-01_2023-01-01.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    columns = list(df.columns)
    close_idx = columns.index('Close') # We use Close to calculate Up/Down
    data_array = df.values 
    
    # Hyperparameters
    SEQ_LENGTH = 60
    INPUT_SIZE = len(columns)
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1         
    LEARNING_RATE = 0.001   
    EPOCHS = 40             
    BATCH_SIZE = 32         
    
    print("Generating Binary (Up/Down) sequences...")
    X, y = create_binary_sequences(data_array, SEQ_LENGTH, close_col_index=close_idx)
    
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = StockClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    
    # NEW MATH: Binary Cross Entropy Loss (The gold standard for Yes/No AI)
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Training Phase (Binary Classifier)...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      
            
            epoch_loss += loss.item()
            
            # Calculate accuracy on the fly
            # If probability > 0.5, we guess "UP" (1)
            predicted_classes = (outputs > 0.5).float()
            correct_predictions += (predicted_classes == batch_y).sum().item()
            total_predictions += batch_y.size(0)
            
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(dataloader)
            accuracy = (correct_predictions / total_predictions) * 100
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_classifier.pth")
    print("\nTraining complete! Model saved to models/lstm_classifier.pth")