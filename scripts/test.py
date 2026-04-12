import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from models.model import Model

X_test = torch.tensor(pd.read_csv("data/X_test.csv").values, dtype=torch.float32)
y_test = pd.read_csv("data/y_test.csv").values.flatten()

model = Model(B=X_test.shape[1])
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():
    preds = model(X_test).argmax(dim=1).numpy()

print("Accuracy:", accuracy_score(y_test, preds))
