"""
Dual-Graph Construction Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DualGraphBuilder(nn.Module):
    """
    Build Speaker Graph and Temporal Graph
    """
    def __init__(self, args):
        super().__init__()
        self.window_past = getattr(args, 'window_past', 3)
        self.window_future = getattr(args, 'window_future', 3)
        
    def forward(self, qmask, umask, seq_lengths):
        """
        Build dual graphs
        Args:
            qmask: [batch, seq_len] Speaker mask (0 or 1 for different speakers)
            umask: [batch, seq_len] Utterance valid mask
            seq_lengths: Length of each conversation
        Returns:
            speaker_graph: Speaker graph adjacency matrix
            temporal_graph: Temporal graph adjacency matrix
        """
        batch_size = qmask.size(0)
        max_seq_len = qmask.size(1)
        device = qmask.device
        
        # Initialize graphs
        speaker_graphs = []
        temporal_graphs = []
        
        for b in range(batch_size):
            seq_len = seq_lengths[b]
            
            # Build Speaker Graph
            speaker_adj = self._build_speaker_graph(
                qmask[b, :seq_len], 
                seq_len, device
            )
            
            # Build Temporal Graph  
            temporal_adj = self._build_temporal_graph(
                seq_len, device
            )
            
            # Padding to max length
            speaker_adj_padded = torch.zeros(max_seq_len, max_seq_len, device=device)
            temporal_adj_padded = torch.zeros(max_seq_len, max_seq_len, device=device)
            
            speaker_adj_padded[:seq_len, :seq_len] = speaker_adj
            temporal_adj_padded[:seq_len, :seq_len] = temporal_adj
            
            speaker_graphs.append(speaker_adj_padded)
            temporal_graphs.append(temporal_adj_padded)
            
        speaker_graph = torch.stack(speaker_graphs)  # [batch, max_seq_len, max_seq_len]
        temporal_graph = torch.stack(temporal_graphs)
        
        return speaker_graph, temporal_graph
    
    def _build_speaker_graph(self, qmask, seq_len, device):
        """
        Build speaker relationship graph (GPU-optimized vectorized version)
        Edge types:
        - self-address: same speaker's own utterances
        - direct addressing: direct dialogue
        - response: response relationship
        """
        # Vectorized implementation - much faster on GPU
        i_idx = torch.arange(seq_len, device=device).view(-1, 1)  # [seq_len, 1]
        j_idx = torch.arange(seq_len, device=device).view(1, -1)  # [1, seq_len]
        
        # Self-connection
        adj = torch.eye(seq_len, device=device)
        
        # Same speaker (self-address) with distance decay
        same_speaker = (qmask.view(-1, 1) == qmask.view(1, -1)).float()
        distance = torch.abs(i_idx - j_idx).float()
        decay_factor = 1.0 / (distance + 1.0)  # Distance-based decay
        same_speaker_weighted = same_speaker * decay_factor * 0.8  # Weighted
        adj = torch.maximum(adj, same_speaker_weighted)  # Take maximum to avoid accumulation
        
        # Different speakers
        diff_speaker = (qmask.view(-1, 1) != qmask.view(1, -1)).float()
        
        # Direct addressing (adjacent utterances)
        distance = torch.abs(i_idx - j_idx)
        direct = (distance == 1).float() * diff_speaker
        adj = torch.maximum(adj, direct)
        
        # Response (within 2 steps, past)
        response_past = ((j_idx < i_idx) & (distance <= 2)).float() * diff_speaker * 0.8
        adj = torch.maximum(adj, response_past)
        
        # Future response
        response_future = ((i_idx < j_idx) & (distance <= 2)).float() * diff_speaker * 0.6
        adj = torch.maximum(adj, response_future)
                        
        return adj
    
    def _build_temporal_graph(self, seq_len, device):
        """
        Build temporal relationship graph (GPU-optimized vectorized version)
        Edge types:
        - past: past utterances with distance decay
        - present: current utterance
        - future: future utterances within window
        """
        i_idx = torch.arange(seq_len, device=device).view(-1, 1)  # [seq_len, 1]
        j_idx = torch.arange(seq_len, device=device).view(1, -1)  # [1, seq_len]
        
        # Self-connection
        adj = torch.eye(seq_len, device=device)
        
        # Distance-based weight: 1 / (distance + 1)
        distance = torch.abs(i_idx - j_idx).float()
        temporal_weight = 1.0 / (distance + 1.0)
        
        # Past: all previous utterances with decay
        past_mask = (j_idx < i_idx).float()
        past_weight = temporal_weight * past_mask
        adj = torch.maximum(adj, past_weight)
        
        # Future: within window with decay
        future_mask = ((j_idx > i_idx) & (j_idx - i_idx <= self.window_future)).float()
        future_weight = temporal_weight * future_mask * 0.8  # Future has slightly lower weight
        adj = torch.maximum(adj, future_weight)
        
        return adj
