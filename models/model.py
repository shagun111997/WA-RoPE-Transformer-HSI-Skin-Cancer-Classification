import torch
import torch.nn as nn
import torch.nn.functional as F

class Tokenizer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(1, d_model)

    def forward(self, x):
        return self.projection(x.unsqueeze(-1))

class WARoPE(nn.Module):
    def __init__(self, B, d_model):
        super().__init__()
        self.position_embeddings = nn.Embedding(B, d_model)

    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        return tokens + self.position_embeddings(positions)

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        return self.ln2(x + self.ffn(x))

class Model(nn.Module):
    def __init__(self, B, d_model=128):
        super().__init__()
        self.tokenizer = Tokenizer(d_model)
        self.rope = WARoPE(B, d_model)
        self.transformer = TransformerBlock(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.rope(x)
        x = self.transformer(x)
        x = x.permute(0,2,1)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
