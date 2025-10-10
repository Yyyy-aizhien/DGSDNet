"""
Score Network Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScoreNetwork(nn.Module):
    """
    Score network based on UNet architecture
    Adapted for spectral space processing
    """
    def __init__(self, hidden_dim, time_embed_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Encoder
        self.encoder = nn.ModuleList([
            SpectralConvBlock(hidden_dim, hidden_dim * 2, time_embed_dim),
            SpectralConvBlock(hidden_dim * 2, hidden_dim * 4, time_embed_dim),
            SpectralConvBlock(hidden_dim * 4, hidden_dim * 8, time_embed_dim),
        ])
        
        # Middle layer
        self.middle = SpectralConvBlock(hidden_dim * 8, hidden_dim * 8, time_embed_dim)
        
        # Decoder
        self.decoder = nn.ModuleList([
            SpectralConvBlock(hidden_dim * 16, hidden_dim * 4, time_embed_dim),
            SpectralConvBlock(hidden_dim * 8, hidden_dim * 2, time_embed_dim),
            SpectralConvBlock(hidden_dim * 4, hidden_dim, time_embed_dim),
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Condition injection
        self.condition_encoder = ConditionEncoder(hidden_dim)
        
    def forward(self, x, eigenvalues, t, condition=None):
        """
        Calculate score function
        Args:
            x: [batch, seq_len, hidden_dim] Input features (in spectral space)
            eigenvalues: [batch, seq_len] Eigenvalues
            t: Time step
            condition: Conditional information (other available modalities)
        """
        # Time embedding
        t_embed = self.time_embedding(t)
        
        # Condition encoding
        if condition is not None:
            cond_embed = self.condition_encoder(condition, eigenvalues)
            x = x + cond_embed  # Condition injection
            
        # Encoding path
        encoder_outputs = []
        h = x
        for encoder_block in self.encoder:
            h = encoder_block(h, t_embed, eigenvalues)
            encoder_outputs.append(h)
            
        # Middle processing
        h = self.middle(h, t_embed, eigenvalues)
        
        # Decoding path (with skip connections)
        for i, decoder_block in enumerate(self.decoder):
            # Skip connections
            skip = encoder_outputs[-(i+1)]
            h = torch.cat([h, skip], dim=-1)
            h = decoder_block(h, t_embed, eigenvalues)
            
        # Output score
        score = self.output_layer(h)
        
        return score

class SpectralConvBlock(nn.Module):
    """Spectral convolution block"""
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        # Normalization
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
            
        # Spectral attention
        self.spectral_attention = SpectralAttention(out_channels)
        
    def forward(self, x, t_embed, eigenvalues=None):
        """
        Forward propagation
        Args:
            x: [batch, seq_len, channels]
            t_embed: [batch, time_embed_dim]
            eigenvalues: [batch, seq_len] Optional eigenvalues
        """
        # Convert dimensions [batch, seq_len, channels] -> [batch, channels, seq_len]
        # If 4D (e.g. [B, T, K, C]), merge K*C into channel dimension
        if x.dim() == 4:
            B, T, K, C = x.shape
            x = x.reshape(B, T, K * C)
        x_trans = x.transpose(1, 2)
        
        # First convolution
        h = self.conv1(x_trans)
        h = h.transpose(1, 2)  # [batch, seq_len, channels]
        h = self.norm1(h)
        
        # Time embedding injection
        t_proj = self.time_proj(t_embed)
        # Unify to [batch, 1, out_channels] then broadcast to [batch, seq_len, out_channels]
        if t_proj.dim() == 1:
            t_proj = t_proj.unsqueeze(0)  # [1, out_channels]
        if t_proj.size(0) == 1:
            t_proj = t_proj.expand(x.size(0), -1)  # [batch, out_channels]
        t_proj = t_proj.unsqueeze(1).expand(x.size(0), x.size(1), -1)
        h = h + t_proj
        h = F.gelu(h)
        
        # Second convolution
        h = h.transpose(1, 2)
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = F.gelu(h)
        
        # Spectral attention (if eigenvalues provided)
        if eigenvalues is not None:
            h = self.spectral_attention(h, eigenvalues)
            
        # Residual connection
        res = self.residual(x_trans).transpose(1, 2)
        h = h + res
        
        return h

class SpectralAttention(nn.Module):
    """Spectral attention mechanism based on eigenvalues"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.scale = 1.0 / np.sqrt(channels)
        
    def forward(self, x, eigenvalues):
        """
        Args:
            x: [batch, seq_len, channels]
            eigenvalues: [batch, seq_len]
        """
        batch_size, seq_len, channels = x.shape
        
        # Q, K, V projection
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Modulate attention using eigenvalues: align with scores[B,T,T]'s last dimension
        # eigenvalues: [B, T] -> eigen_weight: [B, T]
        eigen_weight = torch.sigmoid(eigenvalues)
        # Expand to [B, 1, T], aligning with scores' last dimension T
        scores = scores * eigen_weight.unsqueeze(1)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        output = torch.matmul(attention_weights, V)
        
        return output

class TimeEmbedding(nn.Module):
    """Time step embedding (Gaussian Fourier Features)"""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, t):
        """
        Args:
            t: Scalar or tensor time step
        """
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32)
            
        # Gaussian Fourier features
        t_proj = t.view(-1, 1) * self.W[None, :] * 2 * np.pi
        embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        
        # MLP processing
        embed = self.mlp(embed)
        
        return embed

class ConditionEncoder(nn.Module):
    """Conditional information encoder (for cross-attention)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Ensure hidden_dim is divisible by num_heads
        num_heads = 8
        if hidden_dim % num_heads != 0:
            # Adjust hidden_dim to be divisible by num_heads
            hidden_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Spectral modulation
        self.spectral_modulation = nn.Linear(1, hidden_dim)
        
    def forward(self, condition, eigenvalues=None):
        """
        Encode conditional information
        Args:
            condition: [batch, seq_len, hidden_dim] Conditional features
            eigenvalues: [batch, seq_len] Eigenvalues (optional)
        """
        batch_size, seq_len, hidden_dim = condition.shape
        
        # Cross-modal attention (process condition via self-attention)
        # Auto-correct input shape from [B, H, T] if needed
        if condition.dim() == 3 and eigenvalues is not None:
            B, T = eigenvalues.shape[:2]
            if condition.shape[1] != T and condition.shape[2] == T:
                condition = condition.transpose(1, 2)
        attended, _ = self.cross_attention(
            condition, condition, condition
        )
        
        # Feature fusion
        fused = torch.cat([condition, attended], dim=-1)
        output = self.fusion(fused)
        
        # If eigenvalues provided, apply spectral modulation
        if eigenvalues is not None:
            eigen_mod = self.spectral_modulation(eigenvalues.unsqueeze(-1))
            output = output * torch.sigmoid(eigen_mod)
            
        return output

class ResidualBlock(nn.Module):
    """Residual block"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels)
        )
        
    def forward(self, x):
        return x + self.block(x)
