"""
Dual-Spectral Diffusion Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .score_network import ScoreNetwork

class DualSpectralDiffusion(nn.Module):
    def __init__(self, hidden_dim, num_diffusion_steps=100, sigma=25.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_steps = num_diffusion_steps
        self.sigma = sigma
        
        # Score networks for features and eigenvalues
        self.feature_score_nets = nn.ModuleDict({
            'text_speaker': ScoreNetwork(hidden_dim),
            'text_temporal': ScoreNetwork(hidden_dim),
            'audio_speaker': ScoreNetwork(hidden_dim),
            'audio_temporal': ScoreNetwork(hidden_dim),
            'visual_speaker': ScoreNetwork(hidden_dim),
            'visual_temporal': ScoreNetwork(hidden_dim)
        })
        
        # Score networks for eigenvalues
        self.eigenvalue_score_nets = nn.ModuleDict({
            'speaker': ScoreNetwork(1),  # Eigenvalues are scalars
            'temporal': ScoreNetwork(1)
        })
        
        # Reconstruction modules
        self.reconstruction = nn.ModuleDict({
            'text': nn.Linear(hidden_dim, hidden_dim),
            'audio': nn.Linear(hidden_dim, hidden_dim),
            'visual': nn.Linear(hidden_dim, hidden_dim)
        })
        
    def spectral_decomposition(self, adjacency_matrix):
        """Spectral decomposition of adjacency matrix"""
        # E = U * Λ * U^T
        eigenvalues, eigenvectors = torch.linalg.eigh(adjacency_matrix)
        return eigenvalues, eigenvectors
    
    def marginal_prob_std(self, t):
        """Calculate marginal probability standard deviation"""
        return torch.sqrt((self.sigma**(2*t) - 1.) / (2. * np.log(self.sigma)))
    
    def forward_sde(self, x, t):
        """Forward SDE: dx = σ^t dw"""
        noise = torch.randn_like(x) 
        std = self.marginal_prob_std(t)
        perturbed_x = x + noise * std.view(-1, 1, 1)
        return perturbed_x, noise
    
    def reverse_sde_step(self, x_t, score, t, dt=0.01):
        """Reverse SDE solver"""
        drift = -self.sigma**(2*t) * score
        diffusion = self.sigma**t
        noise = torch.randn_like(x_t)
        x_prev = x_t - drift * dt + diffusion * np.sqrt(dt) * noise
        return x_prev
    
    def forward(self, features, speaker_graph, temporal_graph, missing_mask=None):
        """
        Dual-spectral diffusion process
        Args:
            features: Feature dictionary {'text': [batch, seq_len, hidden_dim], ...}
            speaker_graph: [batch, seq_len, seq_len]
            temporal_graph: [batch, seq_len, seq_len]
            missing_mask: [batch, seq_len, 3] Missing modality mask
        """
        batch_size, seq_len = features['text'].shape[:2]
        device = features['text'].device
        
        # Spectral decomposition of two graphs
        s_eigenvalues, s_eigenvectors = self.spectral_decomposition(speaker_graph)
        t_eigenvalues, t_eigenvectors = self.spectral_decomposition(temporal_graph)
        
        recovered = {}
        
        for modality in ['text', 'audio', 'visual']:
            modality_idx = {'text': 0, 'audio': 1, 'visual': 2}[modality]
            
            if missing_mask is not None and missing_mask[:, :, modality_idx].any():
                # If this modality is missing
                x_T = torch.randn_like(features[modality])
                
                # Reverse diffusion on Speaker graph
                x_s = self._diffuse_on_graph(
                    x_T, modality, 'speaker',
                    s_eigenvalues, s_eigenvectors,
                    features, missing_mask
                )
                
                # Reverse diffusion on Temporal graph  
                x_t = self._diffuse_on_graph(
                    x_T, modality, 'temporal',
                    t_eigenvalues, t_eigenvectors,
                    features, missing_mask
                )
                
                # Fuse results from two paths
                recovered[modality] = (x_s + x_t) / 2
                
                # Reconstruction optimization
                recovered[modality] = self.reconstruction[modality](recovered[modality])
            else:
                recovered[modality] = features[modality]
                
        return recovered
    
    def _diffuse_on_graph(self, x_T, modality, graph_type, eigenvalues, eigenvectors, features, missing_mask):
        """Diffusion on single graph"""
        x = x_T
        device = x.device
        
        # Reverse diffusion process
        for t in torch.linspace(1, 0, self.num_steps, device=device):
            # Calculate feature score
            score_net = self.feature_score_nets[f'{modality}_{graph_type}']
            
            # Use other available modalities as condition
            cond_features = []
            for m in ['text', 'audio', 'visual']:
                if m != modality:
                    modality_idx = {'text': 0, 'audio': 1, 'visual': 2}[m]
                    if missing_mask is None or not missing_mask[:, :, modality_idx].any():
                        cond_features.append(features[m])
            
            if cond_features:
                condition = torch.stack(cond_features, dim=-1).mean(dim=-1)
            else:
                condition = None
                
            # Calculate score in spectral space
            # In spectral space: (U^T @ X), where X is [B, T, H] -> [B, T, H]
            x_spectral = torch.matmul(eigenvectors.transpose(-2, -1), x)
            
            score = score_net(x_spectral, eigenvalues, t, condition)
            
            # Reverse SDE step
            x = self.reverse_sde_step(x, score, t)
            
        return x
