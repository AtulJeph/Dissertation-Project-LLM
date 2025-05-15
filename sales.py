import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv("sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date")
df["Lag1"] = df["Weekly_Sales"].shift(1).fillna(0)
df["MA7"]  = df["Weekly_Sales"].rolling(7).mean().fillna(0)
scaler = MinMaxScaler()
df[["Sales","Lag1","MA7"]] = scaler.fit_transform(df[["Weekly_Sales","Lag1","MA7"]])

class SalesDataset(Dataset):
    def __init__(self, data, window=4):
        X, y = [], []
        for i in range(len(data)-window):
            seq = data[["Sales","Lag1","MA7"]].iloc[i:i+window].values
            X.append(seq)
            y.append(data["Sales"].iloc[i+window])
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

ds = SalesDataset(df, window=4)
dl = DataLoader(ds, batch_size=16, shuffle=True)

class LSTMForecaster(nn.Module):
    def __init__(self, in_dim=3, hid_dim=32, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, batch_first=True)
        self.lin  = nn.Linear(hid_dim, out_dim)
    def forward(self,x):
        out, _ = self.lstm(x)          # (B, T, hid)
        last    = out[:, -1, :]        # (B, hid)
        return self.lin(last)          # (B, 1)

model = LSTMForecaster().to("cuda" if torch.cuda.is_available() else "cpu")
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

model.train()
for Xb, yb in dl:
    Xb, yb = Xb.to(model.lin.weight.device), yb.to(model.lin.weight.device)
    pred   = model(Xb)
    loss   = loss_fn(pred, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
print("Done training numeric forecaster — final loss:", loss.item())

torch.save({"model": model.state_dict(),
            "scaler": scaler}, "sales_forecaster.pt")