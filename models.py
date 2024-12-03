import torch as th
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, ks=(3, 3), s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, ks, s, p)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.PReLU()
        self.mp = nn.MaxPool2d(kernel_size=[2, 1])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.mp(x)
        return x


class CNNEmbedder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        # Changed
        self.layers = nn.Sequential(
            ConvBlock(ch_in, ch_out),
            ConvBlock(ch_out, ch_out),
            ConvBlock(ch_out, ch_out),
            ConvBlock(ch_out, ch_out),
        )
    
    def forward(self, x):
        return self.layers(x)


class VADNet(nn.Module):
    def __init__(self, n_feat=256, cnn_channels=32, embed_dim=256, dff=512, num_heads=16):
        super().__init__()
        self.cnn_embedder = CNNEmbedder(ch_in=1, ch_out=cnn_channels) 
        # after the framewise flattening operation we have F'xC = (n_feat/16)*cnn_channels
        self.fc1 = nn.Linear(in_features=int((n_feat/16)*cnn_channels), out_features=embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True) # changed
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        # Changed
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dff),
            nn.Linear(dff, embed_dim),
        )
        self.fc2 = nn.Linear(embed_dim, 1) # changed
        
    def forward(self, x):
        x = self.cnn_embedder(x)
        x = th.flatten(x, start_dim=1, end_dim=2)
        x = x.permute(0, 2, 1)
        
        # Attention block
        residual = self.fc1(x)
        x, _ = self.self_attention(residual, residual, residual)
        x += residual  # Residual connection
        x = self.layer_norm1(x)

        # MLP block
        residual = x
        x = self.mlp(x)
        x += residual  # Residual connection
        x = self.layer_norm2(x)

        # Final classification
        x = self.fc2(x)
        return x
