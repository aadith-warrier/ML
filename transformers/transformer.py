import torch
import torch.nn as nn

from attention import MultiHeadAttentionLayer

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super().__init__()

        self.attention_layer = MultiHeadAttentionLayer(num_heads=num_heads, 
                                                       embedding_dim=embedding_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, 4*embedding_dim),
                                 nn.GELU(),
                                 nn.Linear(4*embedding_dim, 2*embedding_dim),
                                 nn.GELU(),
                                 nn.Linear(2*embedding_dim, embedding_dim),
                                 nn.GELU())
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        out = x + self.attention_layer(x)
        out = self.layer_norm1(out)

        out = out + self.mlp(out)
        out = self.layer_norm2(out)
        
        return out
    
class Transformer(nn.Module):
    def __init__(self, num_blocks=12):
        super().__init__()

        self.transformer = nn.ModuleList([EncoderBlock(8, 512) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.transformer:
            x = block(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(4, 256, 512)

    encoder_block = EncoderBlock(num_heads=8, embedding_dim=512)
    print(encoder_block(x).shape)

    transformer = Transformer()
    print(transformer(x).shape)