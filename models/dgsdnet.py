"""
DGSDNet Implementation with Ablation Support
Supports component disabling for ablation studies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .spectral_diffusion import DualSpectralDiffusion
from .gated_fusion import GatedSpectralClassification
from .dual_graph import DualGraphBuilder

class DGSDNet(nn.Module):
    def __init__(self, args):
        super(DGSDNet, self).__init__()
        
        # Feature dimensions - use defaults or get from args
        self.d_l = getattr(args, 'text_dim', 768)
        self.d_a = getattr(args, 'audio_dim', 1024)  
        self.d_v = getattr(args, 'video_dim', 512)
        self.hidden_dim = getattr(args, 'hidden_dim', 128)
        self.num_classes = getattr(args, 'num_classes', 4)
        
        # Ablation experiment parameters
        self.disable_speaker_path = getattr(args, 'disable_speaker_path', False)
        self.disable_temporal_path = getattr(args, 'disable_temporal_path', False)
        self.disable_diffusion = getattr(args, 'disable_diffusion', False)
        self.disable_gated_classifier = getattr(args, 'disable_gated_classifier', False)
        
        # Node Construction - 1D convolution for dimension alignment
        kernel_size_l = getattr(args, 'kernel_size_l', 1)
        kernel_size_a = getattr(args, 'kernel_size_a', 1)
        kernel_size_v = getattr(args, 'kernel_size_v', 1)
        
        self.text_conv = nn.Conv1d(self.d_l, self.hidden_dim, kernel_size=kernel_size_l)
        self.audio_conv = nn.Conv1d(self.d_a, self.hidden_dim, kernel_size=kernel_size_a)
        self.video_conv = nn.Conv1d(self.d_v, self.hidden_dim, kernel_size=kernel_size_v)
        
        # Positional encoding
        self.positional_embedding = PositionalEncoding(self.hidden_dim)
        
        # Dual-graph builder (always create, but internal logic handles ablation)
        self.graph_builder = DualGraphBuilder(args)
        
        # Dual-spectral diffusion (unless disabled)
        if not self.disable_diffusion:
            num_diffusion_steps = getattr(args, 'num_diffusion_steps', 100)
            sigma = getattr(args, 'sigma', 25.0)
            
            self.dsd = DualSpectralDiffusion(
                hidden_dim=self.hidden_dim,
                num_diffusion_steps=num_diffusion_steps,
                sigma=sigma
            )
        else:
            self.dsd = None
        
        # Gated spectral classification (unless disabled)
        if not self.disable_gated_classifier:
            alpha = getattr(args, 'alpha', 0.5)  # Graph fusion weight
            self.gsc = GatedSpectralClassification(
                hidden_dim=self.hidden_dim,
                num_classes=self.num_classes,
                alpha=alpha
            )
        else:
            # Simple linear classifier
            self.gsc = nn.Linear(self.hidden_dim * 3, self.num_classes)
    
    def forward(self, text, audio, visual, missing_mask=None, qmask=None, return_auxiliary_losses=False):
        """
        Forward propagation
        Args:
            text: [B, T, d_l] Text features
            audio: [B, T, d_a] Audio features
            visual: [B, T, d_v] Visual features
            missing_mask: [B, T, 3] Missing modality mask
            qmask: [B, T] Speaker mask
            return_auxiliary_losses: If True, return recon_loss and gate_weights
        Returns:
            outputs: [B, T, num_classes] Classification results
            aux_losses: Dict with 'recon_loss' and 'gates' (if return_auxiliary_losses=True)
        """
        batch_size, seq_len = text.shape[:2]
        
        # Create default qmask if not provided
        if qmask is None:
            qmask = torch.zeros(batch_size, seq_len, device=text.device, dtype=torch.long)
        
        # Node Construction
        # Convert dimensions: [B, T, d] -> [B, d, T]
        text_feat = self.text_conv(text.transpose(1, 2)).transpose(1, 2)  # [B, T, hidden_dim]
        audio_feat = self.audio_conv(audio.transpose(1, 2)).transpose(1, 2)  # [B, T, hidden_dim]
        visual_feat = self.video_conv(visual.transpose(1, 2)).transpose(1, 2)  # [B, T, hidden_dim]
        
        # Positional encoding
        text_feat = self.positional_embedding(text_feat)
        audio_feat = self.positional_embedding(audio_feat)
        visual_feat = self.positional_embedding(visual_feat)
        
        # Handle missing modalities
        if missing_mask is not None:
            # For missing modalities: initialize with small noise (DSD will reconstruct)
            # Note: Not complete zeros, but small randomness to help diffusion reconstruction
            noise_scale = 0.01
            if missing_mask[:, :, 0].any():
                text_noise = torch.randn_like(text_feat) * noise_scale
                text_feat = torch.where(missing_mask[:, :, 0:1], text_noise, text_feat)
            if missing_mask[:, :, 1].any():
                audio_noise = torch.randn_like(audio_feat) * noise_scale
                audio_feat = torch.where(missing_mask[:, :, 1:2], audio_noise, audio_feat)
            if missing_mask[:, :, 2].any():
                visual_noise = torch.randn_like(visual_feat) * noise_scale
                visual_feat = torch.where(missing_mask[:, :, 2:3], visual_noise, visual_feat)
        
        # Build node features
        features = {
            'text': text_feat,
            'audio': audio_feat,
            'visual': visual_feat
        }
        
        # Dual-graph construction
        # Use provided qmask, or create default if None
        if qmask is None:
            qmask = torch.zeros(batch_size, seq_len, device=text.device, dtype=torch.long)
        umask = torch.ones(batch_size, seq_len, device=text.device, dtype=torch.bool)
        seq_lengths = torch.full((batch_size,), seq_len, device=text.device, dtype=torch.long)
        speaker_graph, temporal_graph = self.graph_builder(qmask, umask, seq_lengths)
        
        # Dual-spectral diffusion (unless disabled)
        recon_loss = None
        if self.dsd is not None:
            # Reconstruct missing features
            if return_auxiliary_losses:
                reconstructed_features, recon_loss = self.dsd(
                    features=features,
                    speaker_graph=speaker_graph,
                    temporal_graph=temporal_graph,
                    missing_mask=missing_mask,
                    return_loss=True
                )
            else:
                reconstructed_features = self.dsd(
                    features=features,
                    speaker_graph=speaker_graph,
                    temporal_graph=temporal_graph,
                    missing_mask=missing_mask
                )
        else:
            # If no diffusion, use original features directly
            reconstructed_features = features
            recon_loss = torch.tensor(0.0, device=features['text'].device)
        
        # Gated spectral classification (unless disabled)
        gates = None
        if not self.disable_gated_classifier:
            if return_auxiliary_losses:
                outputs, gates = self.gsc(
                    features=reconstructed_features,
                    speaker_graph=speaker_graph,
                    temporal_graph=temporal_graph,
                    qmask=qmask,
                    return_gates=True
                )
            else:
                outputs = self.gsc(
                    features=reconstructed_features,
                    speaker_graph=speaker_graph,
                    temporal_graph=temporal_graph,
                    qmask=qmask
                )
        else:
            # Simple linear classification
            # Concatenate features from three modalities
            combined_features = torch.cat([
                reconstructed_features['text'],
                reconstructed_features['audio'],
                reconstructed_features['visual']
            ], dim=-1)  # [B, T, hidden_dim * 3]
            
            outputs = self.gsc(combined_features)  # [B, T, num_classes]
        
        if return_auxiliary_losses:
            aux_losses = {
                'recon_loss': recon_loss if recon_loss is not None else torch.tensor(0.0, device=text.device),
                'gates': gates
            }
            return outputs, aux_losses
        
        return outputs

class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-np.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, hidden_dim]
        Returns:
            x + pe: [B, T, hidden_dim]
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)
