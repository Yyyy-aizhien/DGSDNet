"""
Gated Spectral Classification Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_graph_conv import SpectralGraphConv

class GatedSpectralClassification(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Entropy-weighted gating
        self.entropy_gate = EntropyWeightedGating(hidden_dim)
        
        # Spectral GNN
        self.spectral_gnn = SpectralGNN(hidden_dim, num_classes)
        
    def forward(self, features, speaker_graph, temporal_graph, qmask):
        """
        Gated spectral classification
        Args:
            features: Feature dictionary {'text': [batch, seq_len, hidden_dim], ...}
            speaker_graph: [batch, seq_len, seq_len]
            temporal_graph: [batch, seq_len, seq_len]
            qmask: [batch, seq_len] Speaker mask
        """
        batch_size, seq_len = features['text'].shape[:2]
        
        # 1. Calculate entropy for each node
        entropy_s = self._compute_entropy(features, speaker_graph)
        entropy_t = self._compute_entropy(features, temporal_graph)
        
        # 2. Entropy-weighted fusion
        fused_features = self.entropy_gate(
            features, 
            speaker_graph, temporal_graph,
            entropy_s, entropy_t, qmask
        )
        
        # 3. Spectral GNN classification
        output = self.spectral_gnn(fused_features, speaker_graph, temporal_graph)
        
        return output
    
    def _compute_entropy(self, features, graph):
        """Calculate node entropy - measure uncertainty of each node feature"""
        entropy = {}
        for modality, feat in features.items():
            # Calculate entropy for each node's feature vector
            # feat: [batch, seq_len, hidden_dim]
            feat_abs = torch.abs(feat)  # Use absolute value to ensure non-negative
            feat_norm = feat_abs / (feat_abs.sum(dim=-1, keepdim=True) + 1e-10)  # Normalize to probability distribution
            # Calculate entropy: -sum(p * log(p))
            entropy[modality] = -torch.sum(feat_norm * torch.log(feat_norm + 1e-10), dim=-1)  # [batch, seq_len]
        return entropy

class EntropyWeightedGating(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Gating weight calculation: concat(x, x, epsilon, {x_j})
        self.gate_weight = nn.Linear(hidden_dim * 3 + 1, hidden_dim)
        
    def forward(self, features, speaker_graph, temporal_graph, entropy_s, entropy_t, qmask):
        """Entropy-based adaptive gated fusion"""
        batch_size, seq_len = features['text'].shape[:2]
        device = features['text'].device
        
        fused = {}
        
        for modality in features.keys():
            feat = features[modality]  # [batch, seq_len, hidden_dim]
            e_s = entropy_s[modality].unsqueeze(-1)  # [batch, seq_len, 1]
            e_t = entropy_t[modality].unsqueeze(-1)  # [batch, seq_len, 1]
            
            # Calculate reference features {x_j} - aggregation of other modalities within same utterance
            ref_features = []
            for m in features.keys():
                if m != modality:
                    ref_features.append(features[m])
            
            if ref_features:
                ref_feat = torch.stack(ref_features, dim=-1).mean(dim=-1)  # [batch, seq_len, hidden_dim]
            else:
                ref_feat = torch.zeros_like(feat)
            
            # Calculate gating weights
            gate_input = torch.cat([feat, feat, ref_feat, e_s], dim=-1)  # [batch, seq_len, 3*hidden_dim+1]
            gate = torch.sigmoid(self.gate_weight(gate_input))  # [batch, seq_len, hidden_dim]
            
            # Weighted fusion
            fused[modality] = gate * feat
            
        return fused

class SpectralGNN(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_layers=3):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        # First layer reduces concatenated 3-modality features to hidden_dim
        self.layers.append(SpectralGraphConv(hidden_dim * 3, hidden_dim))
        # Rest layers maintain hidden_dim
        for _ in range(max(0, num_layers - 1)):
            self.layers.append(SpectralGraphConv(hidden_dim, hidden_dim))
        
        # Classifier: features from 3 modalities
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, features, speaker_graph, temporal_graph):
        """Spectral Graph Neural Network"""
        batch_size, seq_len = features['text'].shape[:2]
        device = features['text'].device
        
        # Fuse adjacency matrices: A = α * E^s + (1-α) * E^q
        alpha = 0.5
        fused_adj = alpha * speaker_graph + (1 - alpha) * temporal_graph
        
        # Add self-connections: A = A + I
        identity = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        fused_adj = fused_adj + identity
        
        # Normalization: D^(-1/2) A D^(-1/2)
        D = torch.sum(fused_adj, dim=-1)  # [batch, seq_len]
        D_sqrt_inv = torch.pow(D + 1e-10, -0.5).unsqueeze(-1)  # [batch, seq_len, 1]
        norm_adj = D_sqrt_inv * fused_adj * D_sqrt_inv.transpose(-2, -1)
        
        # Concatenate all modality features
        h = torch.cat([features['text'], features['audio'], features['visual']], dim=-1)
        
        # GNN propagation
        for layer in self.layers:
            # Process each batch separately
            h_new = []
            for b in range(batch_size):
                h_b = layer(h[b], norm_adj[b])  # [seq_len, hidden_dim*3]
                h_new.append(h_b)
            h = torch.stack(h_new, dim=0)  # [batch, seq_len, hidden_dim*3]
            h = F.relu(h)
            
        # Classification
        output = self.classifier(h)  # [batch, seq_len, num_classes]
        
        return output
