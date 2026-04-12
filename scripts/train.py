import torch
import pandas as pd
from models.model import Model

X_train = torch.tensor(pd.read_csv("data/X_train.csv").values, dtype=torch.float32)
y_train = torch.tensor(pd.read_csv("data/y_train.csv").values.flatten(), dtype=torch.long)

model = Model(B=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    logits = model(X_train)
    loss = torch.nn.functional.cross_entropy(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")
