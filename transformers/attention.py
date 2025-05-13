import math
import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_k = math.sqrt(embedding_dim/num_heads)

        self.W_Q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_K = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_V = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_O = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def forward(self, x):
        B, S, _ = x.size()

        K = self.W_K(x).view(B, S, self.num_heads, -1).permute(0, 2, 1, 3)
        Q = self.W_Q(x).view(B, S, self.num_heads, -1).permute(0, 2, 1, 3)
        V = self.W_V(x).view(B, S, self.num_heads, -1).permute(0, 2, 1, 3)

        attention_weights = torch.nn.functional.softmax(Q@K.mT/self.d_k, dim=-1)
        out = (attention_weights@V).permute(0, 2, 1, 3).reshape(B, S, -1)

        out = self.W_O(out)

        return out

class MaskedMultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_k = math.sqrt(embedding_dim/num_heads)

        self.W_Q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_K = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_V = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_O = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def forward(self, x, mask):
        B, S, _ = x.size()

        K = self.W_K(x).view(B, S, self.num_heads, -1).permute(0, 2, 1, 3)
        Q = self.W_Q(x).view(B, S, self.num_heads, -1).permute(0, 2, 1, 3)
        V = self.W_V(x).view(B, S, self.num_heads, -1).permute(0, 2, 1, 3)

        attention_weights = torch.nn.functional.softmax((Q@K.mT + mask)/self.d_k, dim=-1)
        out = (attention_weights@V).permute(0, 2, 1, 3).reshape(B, S, -1)
        
        out = self.W_O(out)

        return out
    

if __name__ == "__main__": 
    attention_layer = MultiHeadAttentionLayer(num_heads=8, embedding_dim=512)

    x = torch.randn(4, 16, 512) #B, SEQ_LEN, EMBED_DIM
    out = attention_layer(x)

    print(out.shape)