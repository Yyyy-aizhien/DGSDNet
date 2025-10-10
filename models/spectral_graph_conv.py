"""
Spectral Graph Convolution Layer Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralGraphConv(nn.Module):
    """Spectral graph convolution layer"""
    
    def __init__(self, in_features, out_features, bias=True, use_residual=True):
        super(SpectralGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Residual projection (when dimensions don't match)
        if use_residual and in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.residual_proj = None
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        """
        Forward propagation (with residual connection)
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            output: Output features [N, out_features]
        """
        # Graph convolution: H' = D^(-1/2) A D^(-1/2) H W
        support = torch.mm(x, self.weight)  # [N, out_features]
        output = torch.spmm(adj, support)   # [N, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        
        # Residual connection (prevent gradient vanishing)
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            elif self.in_features == self.out_features:
                residual = x
            else:
                residual = None
            
            if residual is not None:
                output = output + residual
            
        return output

class ChebGraphConv(nn.Module):
    """Chebyshev graph convolution layer"""
    
    def __init__(self, in_features, out_features, K=3, bias=True):
        super(ChebGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        
        # Chebyshev polynomial coefficients
        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        """
        Forward propagation
        Args:
            x: Node features [N, in_features]
            adj: Laplacian matrix [N, N]
        Returns:
            output: Output features [N, out_features]
        """
        # Calculate Chebyshev polynomials
        Tx_0 = x  # T_0(x) = x
        Tx_1 = torch.spmm(adj, x)  # T_1(x) = Lx
        
        if self.K == 1:
            Tx = [Tx_0]
        elif self.K == 2:
            Tx = [Tx_0, Tx_1]
        else:
            Tx = [Tx_0, Tx_1]
            for k in range(2, self.K):
                Tx_k = 2 * torch.spmm(adj, Tx_1) - Tx_0
                Tx.append(Tx_k)
                Tx_0, Tx_1 = Tx_1, Tx_k
        
        # Linear combination
        output = torch.zeros_like(Tx[0])
        for k in range(self.K):
            output += torch.mm(Tx[k], self.weight[k])
        
        if self.bias is not None:
            output = output + self.bias
            
        return output

class GraphAttentionConv(nn.Module):
    """Graph attention convolution layer"""
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        """
        Forward propagation
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            output: Output features [N, out_features]
        """
        N = x.size(0)
        
        # Linear transformation
        h = self.W(x)  # [N, out_features]
        
        # Calculate attention coefficients
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), 
                            h.repeat(N, 1)], dim=1)  # [N*N, 2*out_features]
        e = self.leakyrelu(self.a(a_input)).view(N, N)  # [N, N]
        
        # Apply adjacency matrix mask
        e = e.masked_fill(adj == 0, -9e15)
        
        # Calculate attention weights
        attention = F.softmax(e, dim=1)
        attention = self.dropout_layer(attention)
        
        # Aggregate neighbor information
        h_prime = torch.matmul(attention, h)  # [N, out_features]
        
        return h_prime
