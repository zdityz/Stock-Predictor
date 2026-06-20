import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

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

if __name__ == "__main__":
    data_path = "data/processed/multimodal_AAPL_ALPHA_2010-01-01_2023-01-01.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    SEQ_LENGTH = 60
    INPUT_SIZE = len(df.columns) - 1 
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1         
    LEARNING_RATE = 0.001   
    EPOCHS = 40             
    BATCH_SIZE = 32         
    
    print("Generating Stationary sequences...")
    X, y = create_binary_sequences(df, SEQ_LENGTH)
    
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = StockClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Stationary Training Phase...")
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
            
            predicted_classes = (outputs > 0.5).float()
            correct_predictions += (predicted_classes == batch_y).sum().item()
            total_predictions += batch_y.size(0)
            
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(dataloader)
            accuracy = (correct_predictions / total_predictions) * 100
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_classifier.pth")
    print("\nTraining complete. Stationary Model saved to models/lstm_classifier.pth")