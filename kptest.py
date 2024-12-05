import os
import sys
import math
import math
import numbers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from tqdm import tqdm
from typing import List
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt, morphological_gradient, distance_transform_cdt
from skimage.measure import label, regionprops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing, knn_graph
from torch_geometric.utils import to_dense_batch
from torchvision.models import densenet121
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ===========================
# 1. Define Necessary Classes
# ===========================
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from model import UNET, model_vig
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GeometricData
from torch_geometric.nn import GCNConv
import os
from dataset import Dataset as Datasetx
from metrics import *
import pandas as pd
from model import model_smp, model_unet, model_dunet, preprocessing_fn
import segmentation_models_pytorch as smp
from utils import *
import sys
from tqdm import tqdm
from typing import List
from torch.nn import Linear as Lin, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import LayerNorm as LN
import numbers
import math
from torch import Tensor, einsum
from torch import nn
from utils import simplex, one_hot
from scipy.ndimage import distance_transform_edt, morphological_gradient, distance_transform_cdt
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.models import densenet121
from PIL import Image
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import MessagePassing
# Import geomloss for optimal transport
import geomloss  # Optimal Transport library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, einsum
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data as GeometricData
from torch_geometric.nn import (
    MessagePassing,
    GCNConv,
    knn_graph,
    # to_dense_batch,
)
import torchvision
import torchvision.transforms as transforms
from torchvision.models import densenet121

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
import geomloss  # Optimal Transport library

# ===========================
# 1. Define Necessary Classes
# ===========================


class ProjectionHead(nn.Module):
    """
    Projection head to map descriptors to a common dimensional space.
    """
    def __init__(self, input_dim, output_dim=64):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from scipy.optimize import linear_sum_assignment

class RelConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(RelConv, self).__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.root = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x, edge_index):
        self.flow = 'source_to_target'
        out1 = self.propagate(edge_index, x=self.lin1(x))
        self.flow = 'target_to_source'
        out2 = self.propagate(edge_index, x=self.lin2(x))
        return self.root(x) + out1 + out2

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'


class GNNForEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=False, lin=True, dropout=0.0):
        """
        GNN for embedding with optional concatenation of layer outputs.

        Parameters:
        - in_channels (int): Input feature dimension.
        - out_channels (int): Output feature dimension per layer.
        - num_layers (int): Number of GNN layers.
        - batch_norm (bool): Whether to apply batch normalization.
        - cat (bool): Whether to concatenate layer outputs.
        - lin (bool): Whether to apply a final linear layer.
        - dropout (float): Dropout rate.
        """
        super(GNNForEmbedding, self).__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RelConv(in_channels, out_channels))
            if self.batch_norm:
                self.batch_norms.append(nn.LayerNorm(out_channels))
            in_channels = out_channels
        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels
        if self.lin:
            self.out_channels = out_channels
            self.final = nn.Linear(in_channels, out_channels)
        else:
            self.out_channels = in_channels
        self.reset_parameters()

    def reset_parameters(self):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            conv.reset_parameters()
            if self.batch_norm:
                batch_norm.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        xs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(xs[-1], edge_index)
            if self.batch_norm:
                x = self.batch_norms[i](F.relu(x))
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        if self.cat:
            x = torch.cat(xs, dim=-1)
        else:
            x = xs[-1]
        if self.lin:
            x = self.final(x)
        return x


class UpdateCorrespondenceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(UpdateCorrespondenceMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, di):
        out = self.fc1(di)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class KeypointMatchingModel(nn.Module):
    def __init__(self, embedding_dim=64, consensus_dim=64, num_steps=5, detach=True):
        super(KeypointMatchingModel, self).__init__()
        # GNN layers for embeddings
        self.gnn_embed = GNNForEmbedding(
            in_channels=embedding_dim, 
            out_channels=embedding_dim, 
            num_layers=1, 
            batch_norm=True, 
            cat=True,  # Disabled concatenation to fix RuntimeError
            lin=True, 
            dropout=0.2
        )
        self.gnn_consensus = GNNForEmbedding(
            in_channels=embedding_dim, 
            out_channels=embedding_dim, 
            num_layers=1, 
            batch_norm=True, 
            cat=True,  # Disabled concatenation to fix RuntimeError
            lin=True, 
            dropout=0.2
        )
        self.update_mlp = UpdateCorrespondenceMLP(input_dim=consensus_dim)
        self.num_steps = num_steps
        self.detach = detach

    def forward(self, data_s, data_t, k=4):
        """
        Forward pass for matching between source and target graphs.

        Parameters:
        - data_s: Source graph (PyG Data object)
        - data_t: Target graph (PyG Data object)
        - k: Number of top matches to consider

        Returns:
        - S_0: Initial similarity matrix
        - S_L: Final similarity matrix after consensus steps
        - Hs: Embedded source features
        - Ht: Embedded target features
        """
        x_s, edge_index_s, edge_attr_s, batch_s = data_s.x, data_s.edge_index, data_s.edge_weight, data_s.batch
        x_t, edge_index_t, edge_attr_t, batch_t = data_t.x, data_t.edge_index, data_t.edge_weight, data_t.batch
        
        # GNN embeddings
        Hs = self.gnn_embed(x_s, edge_index_s)
        Ht = self.gnn_embed(x_t, edge_index_t)
        
        if self.detach:
            Hs = Hs.detach()
            Ht = Ht.detach()
        
        # Convert to dense batches
        Hs, s_mask = to_dense_batch(Hs, batch_s, fill_value=0)  # Shape: [B, N_s, C]
        Ht, t_mask = to_dense_batch(Ht, batch_t, fill_value=0)  # Shape: [B, N_t, C]
        assert Hs.size(0) == Ht.size(0), 'Encountered unequal batch sizes, graph loader messed up'
        
        B, N_s, C = Hs.size()
        N_t = Ht.size(1)
        
        # Compute attention scores and apply top-k selection
        attention_scores = torch.bmm(Hs, Ht.transpose(1, 2))  # Shape: (B, N_s, N_t)
        k = min(k, N_t)  # Adjust k if N_t is smaller
        topk_scores, topk_indices = torch.topk(attention_scores, k=k, dim=2, largest=True, sorted=True)  # Shape: (B, N_s, k)
        
        # Initialize similarity matrix with -inf and set top-k scores
        S_hat = torch.full_like(attention_scores, float('-inf'))
        S_hat.scatter_(2, topk_indices, topk_scores)
        
        # Apply softmax on the valid top-k entries
        S = F.softmax(S_hat, dim=-1)
        S = S.masked_fill(S_hat == float('-inf'), 0)
        S_0 = S.clone()
        
        for _ in range(self.num_steps):
            # Random features for consensus step
            R_s = torch.randn((B, N_s, C), dtype=Hs.dtype, device=Hs.device)
            
            # Compute target random features via similarity
            R_t = torch.bmm(S.transpose(1, 2), R_s)  # Shape: (B, N_t, C)
            
            # Consensus GNN layers
            R_s_flat = R_s.view(-1, C)
            R_t_flat = R_t.view(-1, C)
            
            # Apply GNN consensus layers
            O_s = self.gnn_consensus(R_s_flat, edge_index_s, edge_attr_s)
            O_t = self.gnn_consensus(R_t_flat, edge_index_t, edge_attr_t)
            
            # Reshape outputs
            O_s = O_s.view(B, N_s, -1)
            O_t = O_t.view(B, N_t, -1)
            
            # Compute differences for top-k pairs
            D = torch.zeros((B, N_s, k, O_s.size(-1)), device=Hs.device)
            for b in range(B):
                indices_t = topk_indices[b]  # Shape: (N_s, k)
                O_t_selected = O_t[b][indices_t]  # Shape: (N_s, k, C_out)
                O_s_expanded = O_s[b].unsqueeze(1).expand(-1, k, -1)  # Shape: (N_s, k, C_out)
                D[b] = O_s_expanded - O_t_selected  # Shape: (N_s, k, C_out)
            
            # Update S_hat for top-k pairs
            delta_S_hat = self.update_mlp(D).squeeze(-1)  # Shape: (B, N_s, k)
            S_hat.scatter_add_(2, topk_indices, delta_S_hat)
            
            # Recompute S with updated S_hat
            S = F.softmax(S_hat, dim=-1)
            S = S.masked_fill(S_hat == float('-inf'), 0)
        
        # Final similarity matrix
        S_L = S.clone()
        return S_0, S_L, Hs, Ht

    def loss_fn(self, S_L, correspondence):
        """
        Compute the loss between the predicted similarity matrix and the ground truth correspondences.
        
        Parameters:
        - S_L: The predicted similarity matrix of shape [B, N_s, N_t]
        - correspondence: Tensor of shape [B, N_s], where correspondence[b, i] is the index of the matching node in the target graph, or -1 if no match.
        
        Returns:
        - loss: Scalar loss value.
        """
        B, N_s, N_t = S_L.shape
        device = S_L.device
        
        # Flatten batch and source nodes
        S_L_flat = S_L.view(B * N_s, N_t)  # Shape: (B*N_s, N_t)
        correspondence_flat = correspondence.view(B * N_s)  # Shape: (B*N_s,)
        
        # Filter out unmatched nodes
        valid = correspondence_flat >= 0
        S_L_valid = S_L_flat[valid]  # Shape: (M, N_t)
        target = correspondence_flat[valid]  # Shape: (M,)
        
        if target.numel() == 0:
            return torch.tensor(0.0, device=device)
        
        # Compute Cross-Entropy Loss
        loss = F.cross_entropy(S_L_valid, target)
        return loss

    def acc(self, S, y, reduction='sum'):
        S_np = S.cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(-S_np)
        pred = torch.full_like(y, -1)
        for r, c in zip(row_ind, col_ind):
            if r < S.size(0) and c < S.size(1):
                pred[r] = c
        correct = (pred == y).sum().float()
        if reduction == 'mean':
            return correct / y.size(0)
        return correct

class LearnableFeatureMapper(nn.Module):
    def __init__(self, feature_dim, num_bins=10):
        super().__init__()
        # Learnable projection to scalar
        self.scalar_proj = nn.Linear(feature_dim, 1)
        
        # Learnable projection to bins
        self.projection = nn.Linear(feature_dim, num_bins)
        
        # Optional: additional feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, features):
        # Transform features
        features_transformed = self.feature_transform(features)
        
        # Project to scalar
        scalar_features = self.scalar_proj(features_transformed)  # Shape: (N, 1)
        
        # Project to bins
        bin_logits = self.projection(features_transformed)       # Shape: (N, num_bins)
        bin_probabilities = F.softmax(bin_logits, dim=-1)
        
        return scalar_features, bin_probabilities


# ===========================
# 2. Define Loss Functions
# ===========================

def descriptor_consistency_loss(descriptors1, descriptors2, config={}, mask=None, batch_size=1, H=64, W=64):
    """
    Encourage descriptors from different layers or warped images to be consistent by 
    computing per-descriptor similarity and applying margin-based loss.

    This function can handle descriptors provided in two formats:
    - [N, C], where N = batch_size * H * W
    - [B, C, H, W]

    Parameters:
    - descriptors1 (torch.Tensor): Descriptors from layer1, shape [N, C] or [B, C, H, W]
    - descriptors2 (torch.Tensor): Descriptors from layer2 (warped), shape [N, C] or [B, C, H, W]
    - config (dict): Configuration dictionary containing:
        - 'positive_margin' (float): Margin for positive distances
        - 'negative_margin' (float): Margin for negative distances
        - 'lambda_d' (float): Weight for positive loss
    - mask (torch.Tensor, optional): Mask to consider valid regions, shape [B, H, W]
    - batch_size (int, optional): Number of samples in the batch (default: 1)
    - H (int, optional): Height of the descriptor map (default: 64)
    - W (int, optional): Width of the descriptor map (default: 64)

    Returns:
    - loss (torch.Tensor): Scalar loss value.
    """
    # Determine input shape and reshape if necessary
    if descriptors1.dim() == 2 and descriptors2.dim() == 2:
        # Input shape: [N, C] where N = B * H * W
        N, C = descriptors1.shape
        assert N == batch_size * H * W, f"Expected N={batch_size * H * W}, but got N={N}"
        
        # Reshape to [B, C, H, W]
        descriptors1 = descriptors1.view(batch_size, C, H, W)
        descriptors2 = descriptors2.view(batch_size, C, H, W)
   
    elif descriptors1.dim() == 4 and descriptors2.dim() == 4:
        # Input shape: [B, C, H, W]
        B1, C1, H1, W1 = descriptors1.shape
        B2, C2, H2, W2 = descriptors2.shape
        assert B1 == B2 == batch_size, f"Batch sizes do not match: {B1} vs {B2}"
        assert C1 == C2 == descriptors1.shape[1], f"Channel sizes do not match: {C1} vs {C2}"
        assert H1 == H2 == H and W1 == W2 == W, f"Spatial dimensions do not match: ({H1}, {W1}) vs ({H2}, {W2})"
    else:
        raise ValueError("Descriptors must be either [N, C] or [B, C, H, W] tensors.")

    # Normalize descriptors along the channel dimension
    descriptors1 = F.normalize(descriptors1, p=2, dim=1)  # [B, C, H, W]
    descriptors2 = F.normalize(descriptors2, p=2, dim=1)  # [B, C, H, W]

    # Compute cosine similarity per descriptor: [B, H, W]
    similarity = torch.sum(descriptors1 * descriptors2, dim=1)  # [B, H, W]

    # Optionally apply ReLU to ensure non-negative similarity
    similarity = F.relu(similarity)  # [B, H, W]

    # Retrieve configuration parameters with defaults
    positive_margin = config.get('positive_margin', 1.0)
    negative_margin = config.get('negative_margin', 0.2)
    lambda_d = config.get('lambda_d', 0.5)

    # Define 's' based on provided mask or similarity threshold
    if mask is not None:
        # Assuming mask indicates positive samples (s=1) and negative samples (s=0)
        # Ensure mask is of type float
        s = mask.float()  # [B, H, W]
    else:
        # Define s based on similarity threshold, e.g., s=1 if similarity > 0.5
        s = (similarity > 0.5).float()  # [B, H, W]

    # Compute positive and negative distances
    positive_dist = torch.clamp(positive_margin - similarity, min=0.0)  # [B, H, W]
    negative_dist = torch.clamp(similarity - negative_margin, min=0.0)  # [B, H, W]

    # Compute the weighted loss
    loss = lambda_d * s * positive_dist + (1 - s) * negative_dist  # [B, H, W]

    # Average the loss over all descriptors
    loss = loss.mean()

    return loss


def feature_mapping_loss(scalar_features, bin_probabilities, num_bins=10):
    """
    Compute the feature mapping loss combining OT loss and entropy regularization.

    Parameters:
    - scalar_features (torch.Tensor): Scalar projections, shape [N, 1]
    - bin_probabilities (torch.Tensor): Bin probabilities, shape [N, num_bins]
    - num_bins (int): Number of bins

    Returns:
    - loss (torch.Tensor): Combined loss value
    """
    # Squeeze scalar_features to remove the singleton dimension
    scalar_features = scalar_features.squeeze(1)  # Shape: (N,)
    
    # Compute expected bin positions
    bin_indices = torch.linspace(0, 1, num_bins, device=scalar_features.device)  # Shape: (num_bins,)
    expected_bin_positions = (bin_probabilities * bin_indices.view(1, -1)).sum(dim=1)  # Shape: (N,)
    
    # Compute element-wise distances between features and expected positions
    distances = (scalar_features - expected_bin_positions) ** 2  # Shape: (N,)
    
    # Compute the loss as the mean of distances
    ot_loss = distances.mean()
    
    # Entropy loss to encourage spread of features across bins
    entropy_loss = -torch.mean(bin_probabilities * torch.log(bin_probabilities + 1e-8))
    
    return ot_loss + 0.1 * entropy_loss  # Adjust the weighting as needed

# ===========================
# 3. Define Utility Functions
# ===========================

def sinkhorn_normalization(S, num_iters=10, epsilon=1e-9):
    S = torch.exp(S)
    for i in range(num_iters):
        S = S / (S.sum(dim=1, keepdim=True) + epsilon)
        S = S / (S.sum(dim=0, keepdim=True) + epsilon)
        if torch.isnan(S).any():
            print(f"Sinkhorn iteration {i}: NaN encountered in S.")
            S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return S


def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out


def to_sparse(x, mask):
    return x[mask]


def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out


def compute_node_correspondences(positions1, labels1, positions2, labels2, threshold=None):
    """
    Compute correspondences between nodes of two graphs based on labels and positions.

    Parameters:
    - positions1: Tensor of shape (N1, 2), positions of nodes in graph 1
    - labels1: Tensor of shape (N1,), labels of nodes in graph 1
    - positions2: Tensor of shape (N2, 2), positions of nodes in graph 2
    - labels2: Tensor of shape (N2,), labels of nodes in graph 2
    - threshold: Optional float, maximum distance to consider a valid match

    Returns:
    - correspondence: Tensor of shape (N1,), mapping from nodes in graph 1 to nodes in graph 2
                      If a node has no valid match, correspondence[i] = -1
    """
    N1 = positions1.shape[0]
    device = positions1.device  # Get the device ('cuda:0' or 'cpu')
    correspondence = torch.full((N1,), -1, dtype=torch.long, device=device)

    unique_labels = torch.unique(labels1)
    for label in unique_labels:
        # Get indices of nodes with the current label in both graphs
        idx1 = (labels1 == label).nonzero(as_tuple=False).view(-1)
        idx2 = (labels2 == label).nonzero(as_tuple=False).view(-1)

        if idx1.numel() == 0 or idx2.numel() == 0:
            continue

        # Get positions of nodes with the current label
        pos1 = positions1[idx1]  # Shape: (M1, 2)
        pos2 = positions2[idx2]  # Shape: (M2, 2)

        # Compute cost matrix based on positions
        pos1_np = pos1.cpu().numpy()
        pos2_np = pos2.cpu().numpy()
        cost_matrix = np.linalg.norm(pos1_np[:, np.newaxis, :] - pos2_np[np.newaxis, :, :], axis=2)  # Shape: (M1, M2)

        # Apply threshold if provided
        if threshold is not None:
            cost_matrix[cost_matrix > threshold] = 1e6  # Assign a high cost to distances above the threshold

        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter matches based on threshold
        if threshold is not None:
            valid_matches = cost_matrix[row_ind, col_ind] <= threshold
            row_ind = row_ind[valid_matches]
            col_ind = col_ind[valid_matches]

        # Update correspondence
        # Ensure idx1 and idx2 are on the same device as correspondence
        idx1_device = idx1[row_ind].to(device)
        idx2_device = idx2[col_ind].to(device)
        correspondence[idx1_device] = idx2_device

    return correspondence


def visualize_correspondences(img1, img2, kp1, kp2, correspondence, epoch, sample_idx, RESIZE=(224, 224)):
    img1 = cv2.resize(img1, RESIZE)
    img2 = cv2.resize(img2, RESIZE)
    kp1 = np.array(kp1, dtype=int)
    kp2 = np.array(kp2, dtype=int)
    combined_image = np.hstack((img1, img2))
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    for i, j in enumerate(correspondence):
        if j != -1 and i < len(kp1) and j < len(kp2):
            x1, y1 = kp1[i]
            x2, y2 = kp2[j]
            x2 += img1.shape[1]
            plt.plot(x1, y1, 'ro')
            plt.plot(x2, y2, 'bo')
            plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1)
    plt.title(f"Epoch {epoch+1} Sample {sample_idx} Keypoint Correspondences")
    plt.show()


def propagate_labels(graph):
    """
    Propagate labels to unmatched nodes based on their neighbors, 
    using weighted label propagation.

    Parameters:
    - graph: PyG Data object with attributes x, edge_index, y, edge_weight

    Returns:
    - graph: Updated graph with propagated labels
    """
    device = graph.x.device

    # Ensure all necessary tensors are on the same device
    graph.y = graph.y.to(device)
    graph.edge_index = graph.edge_index.to(device)
    
    # Check if edge_weight exists, if not create uniform weights
    if not hasattr(graph, 'edge_weight') or graph.edge_weight is None:
        graph.edge_weight = torch.ones(graph.edge_index.size(1), device=device)
    else:
        graph.edge_weight = graph.edge_weight.to(device)

    # Get node labels and identify unmatched nodes
    labels = graph.y.clone()
    unmatched_nodes = (labels == -1).nonzero(as_tuple=False).view(-1)

    # Create a sparse adjacency matrix with weighted edges
    adj = torch.sparse_coo_tensor(
        graph.edge_index, 
        graph.edge_weight, 
        (graph.num_nodes, graph.num_nodes), 
        device=device
    )

    # Propagate labels for each unmatched node
    for node in unmatched_nodes:
        # Get neighbors and their corresponding edge weights
        adj_row = adj[node].coalesce()
        neighbors = adj_row.indices()[0]
        neighbor_weights = adj_row.values()
        
        # Get labels of neighbors
        neighbor_labels = labels[neighbors]
        
        # Filter out unmatched neighbors
        valid_mask = neighbor_labels != -1
        valid_neighbors = neighbors[valid_mask]
        valid_weights = neighbor_weights[valid_mask]
        valid_neighbor_labels = neighbor_labels[valid_mask]

        if valid_neighbors.numel() > 0:
            # Weighted label assignment
            unique_labels, label_counts = torch.unique(valid_neighbor_labels, return_counts=True)
            weighted_label_scores = torch.zeros_like(unique_labels, dtype=torch.float)

            for i, label in enumerate(unique_labels):
                # Sum weights for each unique label
                label_mask = valid_neighbor_labels == label
                weighted_label_scores[i] = torch.sum(valid_weights[label_mask])

            # Select the label with the highest cumulative weight
            most_common_label = unique_labels[weighted_label_scores.argmax()].item()
            labels[node] = most_common_label
        else:
            # Keep label as -1 if no valid neighbors
            pass

    # Update graph labels
    graph.y = labels
    return graph


def create_graph(features_per_layer, positions, layer_names, device, H, W):
    """
    Create graphs for each semantic subgraph based on feature mappings.

    Parameters:
    - features_per_layer: Dict containing features per layer.
    - positions: Tensor of shape (N, 2), positions of all pixels.
    - layer_names: List of layer names.
    - device: Device to place tensors on.
    - H: Height of the image.
    - W: Width of the image.

    Returns:
    - per_image_graphs: List of PyG Data objects representing graphs.
    """
    per_image_graphs = []

    for layer_name in layer_names:
        if layer_name not in features_per_layer:
            print(f"Layer {layer_name} not in features_per_layer")
            continue

        features = features_per_layer[layer_name]  # Shape: [N, C_in]
        N = H * W

        # Assume node_labels are assigned based on bin assignments or semantic classes
        # Here, we'll mock this as class labels for demonstration
        # Replace with actual label assignments
        node_labels = torch.randint(0, 10, (N,), device=device)  # Shape: (N,)

        # Create intra-subgraph edges (connect pixels within the same subgraph using 8-connected neighborhoods)
        intra_edge_indices = []
        pos_to_idx = {}
        for idx in range(N):
            x, y = idx % W, idx // W
            pos_to_idx[(x, y)] = idx

        for idx in range(N):
            x, y = idx % W, idx // W
            current_label = node_labels[idx].item()
            neighbors = [
                (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
                (x, y - 1),             (x, y + 1),
                (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
            ]
            for nx, ny in neighbors:
                if 0 <= nx < W and 0 <= ny < H:
                    neighbor_label = node_labels[ny * W + nx].item()
                    if neighbor_label == current_label:
                        intra_edge_indices.append([idx, ny * W + nx])

        if intra_edge_indices:
            intra_edge_index = torch.tensor(intra_edge_indices, dtype=torch.long).t().contiguous().to(device)
        else:
            intra_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

        # Create inter-subgraph edges (connect different subgraphs via k-NN of mean features)
        unique_labels = torch.unique(node_labels)
        subgraph_means = []
        subgraph_indices_list = []

        for label in unique_labels:
            subgraph_mask = (node_labels == label)
            if subgraph_mask.sum() == 0:
                continue
            subgraph_indices = subgraph_mask.nonzero(as_tuple=False).view(-1)
            sub_features = features[subgraph_indices]
            mean_feature = sub_features.mean(dim=0)  # Shape: (C_in,)
            mean_position = positions[subgraph_indices].float().mean(dim=0)  # Shape: (2,)
            subgraph_means.append({
                'label': label.item(),
                'mean_feature': mean_feature,
                'mean_position': mean_position
            })
            subgraph_indices_list.append(label.item())

        # Create subgraph-level nodes for k-NN
        if len(subgraph_means) > 0:
            subgraph_features = torch.stack([s['mean_feature'] for s in subgraph_means])  # Shape: (num_subgraphs, C_in)
            subgraph_positions = torch.stack([s['mean_position'] for s in subgraph_means])  # Shape: (num_subgraphs, 2)

            # Perform k-NN on mean features
            num_subgraphs = len(subgraph_features)
            k = min(4, num_subgraphs - 1)  # Adjust k
            if k > 0:
                edge_index_sub = knn_graph(subgraph_features, k=k, batch=None, loop=False)  # Shape: [2, num_edges]

                # Map subgraph-level edges back to node-level edges
                inter_edge_indices = []
                for src_sub_idx, dst_sub_idx in edge_index_sub.t().tolist():
                    src_label = subgraph_indices_list[src_sub_idx]
                    dst_label = subgraph_indices_list[dst_sub_idx]
                    # Get nodes belonging to these subgraphs
                    src_nodes = (node_labels == src_label).nonzero(as_tuple=False).view(-1)
                    dst_nodes = (node_labels == dst_label).nonzero(as_tuple=False).view(-1)
                    num_samples = min(10, src_nodes.numel(), dst_nodes.numel())  # Adjust as needed
                    if num_samples > 0:
                        src_sample = src_nodes[torch.randperm(src_nodes.numel())[:num_samples]]
                        dst_sample = dst_nodes[torch.randperm(dst_nodes.numel())[:num_samples]]
                        for src_node in src_sample:
                            for dst_node in dst_sample:
                                inter_edge_indices.append([src_node.item(), dst_node.item()])

                if inter_edge_indices:
                    inter_edge_index = torch.tensor(inter_edge_indices, dtype=torch.long).t().contiguous().to(device)
                else:
                    inter_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            else:
                inter_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        else:
            inter_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

        # Combine intra and inter edges
        edge_index = torch.cat([intra_edge_index, inter_edge_index], dim=1)
        edge_weight = compute_edge_weights(positions, edge_index)

        # Create PyG Data object
        data = GeometricData(x=features, edge_index=edge_index, edge_weight=edge_weight)
        data.y = node_labels            # Assign labels (indices from 1D map)
        data.pos = positions            # Assign positions
        data.layer = layer_name         # Layer identifier

        # Append to the list of graphs
        per_image_graphs.append(data)

    return per_image_graphs


def compute_edge_weights(positions, edge_index):
    """
    Compute edge weights based on pixel positions for the given edges.
    Closer pixels have higher weights, farther pixels have lower weights.
    """
    # Positions of source and target nodes
    pos_i = positions[edge_index[0]]  # Shape: (num_edges, 2)
    pos_j = positions[edge_index[1]]  # Shape: (num_edges, 2)
    
    # Compute Euclidean distances between connected nodes
    distances = torch.norm(pos_i - pos_j, dim=1)  # Shape: (num_edges,)
    
    # Convert distances to weights (inverse relationship)
    # Use an exponential decay to create weight falloff
    # Avoid division by zero in case distances.mean() is zero
    mean_distance = distances.mean() + 1e-8
    weights = torch.exp(-distances / (2 * mean_distance))
    
    return weights


# ===========================
# 4. Initialize Models and Components
# ===========================

# Dataset directories
DATA_DIR = 'Dataset_1000'
this_dir_path = os.path.abspath(os.getcwd())

x_train_dir = os.path.join(DATA_DIR , 'images')
y_train_dir = os.path.join(DATA_DIR , 'masks')
x_valid_dir = os.path.join(DATA_DIR, 'rgb252')
y_valid_dir = os.path.join(DATA_DIR, 'mask252')
x_test_dir = os.path.join(DATA_DIR , 'test50_rgb')
y_test_dir = os.path.join(DATA_DIR, 'test50_mask')

# Print dataset sizes
print(len(os.listdir(x_train_dir)))
print(len(os.listdir(y_train_dir)))
print(len(os.listdir(x_valid_dir)))
print(len(os.listdir(y_valid_dir)))
print(len(os.listdir(x_test_dir)))
print(len(os.listdir(y_test_dir)))

print(x_test_dir)

# Class dictionary
class_dict = pd.read_csv("label_class_dict.csv")
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()
class_rgb_dict = {cls_name: rgb for cls_name, rgb in zip(class_names, class_rgb_values)}

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)

# Define classes and colors
CLASSES = [ 'background','road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']

class_colors_bgr = [
    [255, 0, 255],   # road
    [255, 0, 0],     # lanemarks
    [0, 255, 0],     # curb
    [0, 0, 255],     # person
    [255, 255, 255], # rider
    [255, 255, 0],   # vehicles
    [0, 255, 255],   # bicycle
    [128, 128, 255], # motorcycle
    [0, 128, 128]    # traffic sign
]

# Convert BGR to RGB
class_colors_rgb = [list(reversed(color)) for color in class_colors_bgr]

# Training parameters
NUM_KEYPOINTS = 20
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
SHIFT = (1, 1)
RESIZE = (224, 224)
SPARSITY_K = 1
CONSENSUS_ITERATIONS = 10
RANDOM_FUNCTIONS = 3
CORRESPONDENCE_THRESHOLD = None
LAMBDA_REG = 0.2


# ===========================
# 4. Initialize Models and Components
# ===========================

# Dataset directories
DATA_DIR = 'Dataset_1000'
this_dir_path = os.path.abspath(os.getcwd())

x_train_dir = os.path.join(DATA_DIR , 'images')
y_train_dir = os.path.join(DATA_DIR , 'masks')
x_valid_dir = os.path.join(DATA_DIR, 'rgb252')
y_valid_dir = os.path.join(DATA_DIR, 'mask252')
x_test_dir = os.path.join(DATA_DIR , 'test50_rgb')
y_test_dir = os.path.join(DATA_DIR, 'test50_mask')

# Print dataset sizes
print(len(os.listdir(x_train_dir)))
print(len(os.listdir(y_train_dir)))
print(len(os.listdir(x_valid_dir)))
print(len(os.listdir(y_valid_dir)))
print(len(os.listdir(x_test_dir)))
print(len(os.listdir(y_test_dir)))

print(x_test_dir)

# Class dictionary
class_dict = pd.read_csv("label_class_dict.csv")
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()
class_rgb_dict = {cls_name: rgb for cls_name, rgb in zip(class_names, class_rgb_values)}

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)

# Define classes and colors
CLASSES = [ 'background','road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']

class_colors_bgr = [
    [255, 0, 255],   # road
    [255, 0, 0],     # lanemarks
    [0, 255, 0],     # curb
    [0, 0, 255],     # person
    [255, 255, 255], # rider
    [255, 255, 0],   # vehicles
    [0, 255, 255],   # bicycle
    [128, 128, 255], # motorcycle
    [0, 128, 128]    # traffic sign
]

# Convert BGR to RGB
class_colors_rgb = [list(reversed(color)) for color in class_colors_bgr]

# Training parameters
NUM_KEYPOINTS = 20
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
SHIFT = (1, 1)
RESIZE = (224, 224)
SPARSITY_K = 1
CONSENSUS_ITERATIONS = 10
RANDOM_FUNCTIONS = 3
CORRESPONDENCE_THRESHOLD = None
LAMBDA_REG = 0.2

# Initialize datasets
train_dataset = Datasetx(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Datasetx(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

class_indices = {class_name: index for index, class_name in enumerate(CLASSES)}

classes_to_consider = ['background','road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']
print(class_indices)
idc = [class_indices[class_name] for class_name in classes_to_consider]




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Backbone Model (DenseNet121) and set to evaluation mode
backbone_model = densenet121(pretrained=True).to(device)
backbone_model.eval()  # Set model to evaluation mode

# Define layers to extract features from
layer_names = ['conv0', 'denseblock1', 'denseblock2', 'denseblock3']
layers = [
    backbone_model.features.conv0,
    backbone_model.features.denseblock1,
    backbone_model.features.denseblock2,
    backbone_model.features.denseblock3,
]

# Dictionary to store activation maps
activation_maps = {}

# Function to get activation maps
def get_activation(name):
    def hook(model, input, output):
        activation_maps[name] = output.detach()
    return hook

# Register hooks to layers
for name, layer in zip(layer_names, layers):
    layer.register_forward_hook(get_activation(name))

# Initialize Projection Heads
layers_to_project = ['conv0', 'denseblock1', 'denseblock2', 'denseblock3']
input_feature_dims = [64, 256, 512, 1024]
projection_heads = {}
common_dim = 64  # Output dimension after projection

for layer_name, input_dim in zip(layers_to_project, input_feature_dims):
    projection_head = ProjectionHead(input_dim=input_dim, output_dim=common_dim).to(device)
    projection_heads[layer_name] = projection_head

# Initialize Feature Mappings
feature_mappings = {}
for layer_name in layer_names:
    feature_mapping = LearnableFeatureMapper(feature_dim=common_dim, num_bins=10).to(device)
    feature_mappings[layer_name] = feature_mapping

# Initialize KeypointMatchingModel
input_feature_dims_for_model = [common_dim for _ in layer_names]  # [64, 64, 64, 64]
modelk = KeypointMatchingModel(
    embedding_dim=common_dim,
    consensus_dim=common_dim,
    num_steps=5,
    detach=True
).to(device)

# Initialize Segmentation Head
segmentation_head = nn.Conv2d(in_channels=common_dim, out_channels=len(CLASSES), kernel_size=1).to(device)

# Collect all parameters to be optimized
all_parameters = list(modelk.parameters()) + list(segmentation_head.parameters())
for fm in feature_mappings.values():
    all_parameters += list(fm.parameters())
for ph in projection_heads.values():
    all_parameters += list(ph.parameters())

# Initialize Optimizer
optimizer = optim.Adam(all_parameters, lr=LEARNING_RATE)

# Define Loss Function for segmentation
criterion = nn.CrossEntropyLoss()


# ===========================
# 5. Training Loop
# ===========================
# ===========================
# 5. Corrected Training Loop
# ===========================

for epoch in range(NUM_EPOCHS):
    modelk.train()
    segmentation_head.train()
    for fm in feature_mappings.values():
        fm.train()
    for ph in projection_heads.values():
        ph.train()
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        B, _, H, W = images.shape  # Batch size and image dimensions

        # Reset activation maps for this batch
        activation_maps = {}

        # Forward pass through the backbone model to populate activation_maps via hooks
        with torch.no_grad():
            _ = backbone_model(images)  # Outputs are captured by hooks

        # Check if activation_maps is populated
        if not activation_maps:
            print("Activation maps are empty. Ensure that hooks are correctly set.")
            continue

        # Initialize losses
        batch_loss_mapping = torch.tensor(0.0, device=device)
        batch_loss_matching = torch.tensor(0.0, device=device)
        batch_loss_segmentation = torch.tensor(0.0, device=device)
        batch_loss_consistency = torch.tensor(0.0, device=device)

        optimizer.zero_grad()

        # Process each image in the batch
        for b in range(B):
            per_image_graphs = []

            # Extract features from all layers and apply projection heads
            features_per_layer = {}
            projected_features_per_layer = {}
            for layer_name in layer_names:
                if layer_name not in activation_maps:
                    print(f"Layer {layer_name} not in activation_maps")
                    continue

                features = activation_maps[layer_name][b].unsqueeze(0)  # Shape: (1, C, H', W')
                _, C, H_feat, W_feat = features.shape

                # Upsample activation maps to match input image size
                features_upsampled = F.interpolate(
                    features,
                    size=(H, W),  # Match input image size
                    mode='bilinear',
                    align_corners=False
                )  # Shape: (1, C, H, W)

                features_b = features_upsampled[0]  # Shape: (C, H, W)

                # Flatten features
                features_flat = features_b.view(C, -1).t()  # Shape: (N, C), N = H * W

                # Store raw features if needed for consistency loss
                features_per_layer[layer_name] = features_flat  # Raw features

                # Apply projection head
                projection_head = projection_heads[layer_name]
                projected = projection_head(features_flat)      # Shape: [N, 64]
                projected_features_per_layer[layer_name] = projected

                # Compute descriptor consistency loss between current layer and all previous layers
                for prev_layer in layers_to_project:
                    if prev_layer == layer_name:
                        continue
                    if prev_layer in projected_features_per_layer:
                        descriptors1 = projected_features_per_layer[prev_layer]  # Shape: [N, 64]
                        descriptors2 = projected_features_per_layer[layer_name]  # Shape: [N, 64]
                        loss = descriptor_consistency_loss(descriptors1, descriptors2)
                        batch_loss_consistency += loss
                # Note: Alternatively, compute consistency across all unique layer pairs outside the loop

            # Feature Mapping and Feature Mapping Loss
            # Apply LearnableFeatureMapper to projected features
            for layer_name, projected in projected_features_per_layer.items():
                feature_mapper = feature_mappings[layer_name]
                scalar_features, bin_probabilities = feature_mapper(projected)  # scalar_features: (N, 1), bin_probabilities: (N, num_bins)
                loss_mapping = feature_mapping_loss(scalar_features, bin_probabilities, num_bins=10)
                batch_loss_mapping += loss_mapping

            # Create graphs for each layer using projected features
            for layer_name, projected in projected_features_per_layer.items():
                # Get positions for the current image
                x_coords = torch.arange(W, device=device).repeat(H, 1).view(-1)  # Shape: (N,)
                y_coords = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).view(-1)  # Shape: (N,)
                positions = torch.stack([x_coords, y_coords], dim=1).float()  # Shape: (N, 2)

                # Create graph for the current layer using projected features
                graphs = create_graph(
                    features_per_layer={layer_name: projected},  # Pass projected features
                    positions=positions,
                    layer_names=[layer_name],
                    device=device,
                    H=H,
                    W=W
                )
                per_image_graphs.extend(graphs)

            # Perform graph consensus matching
            # Assuming per_image_graphs contains graphs from different layers
            # Here, define how to pair graphs for matching (e.g., consecutive layers)
            # For demonstration, we'll match each layer with the next one

            for idx in range(len(per_image_graphs) - 1):
                g_s = per_image_graphs[idx]       # Source graph
                g_t = per_image_graphs[idx + 1]   # Target graph

                # Perform matching between g_s and g_t
                S_0, S_L, Hs, Ht = modelk(g_s, g_t, k=4)  # Updated forward method expects (data_s, data_t, k=4)

                # Compute correspondences
                correspondence = compute_node_correspondences(g_s.pos, g_s.y, g_t.pos, g_t.y, threshold=10.0)

                if torch.any(correspondence != -1):
                    # Compute matching loss
                    loss_matching = modelk.loss_fn(S_L, correspondence)

                    # Regularization term
                    reg_loss = torch.mean((S_L.sum(dim=-1) - 1) ** 2)
                    loss_matching = loss_matching + LAMBDA_REG * reg_loss  # Using LAMBDA_REG=0.2
                    batch_loss_matching += loss_matching

                    # Update node embeddings in g_t
                    valid_matches = correspondence != -1
                    device_match = correspondence.device
                    indices_s = torch.arange(correspondence.size(0), device=device_match)[valid_matches]
                    indices_t = correspondence[valid_matches]

                    # Update node embeddings in g_t
                    Hs_flat = Hs.view(-1, Hs.size(-1))  # Shape: [B*N_s, C]
                    Ht_flat = Ht.view(-1, Ht.size(-1))  # Shape: [B*N_t, C]
                    x_s_enc = Hs_flat[indices_s]
                    x_t_enc = Ht_flat[indices_t]
                    Ht_flat[indices_t] = (Ht_flat[indices_t] + x_s_enc) / 2.0
                    g_t.x = Ht_flat.view(B, -1, Ht_flat.size(-1))[0]

                    # Update labels in g_t based on matched nodes
                    matched_labels = g_s.y[indices_s]
                    g_t.y[indices_t] = matched_labels

                else:
                    print(f"No valid correspondences found between Graph {idx}-{idx + 1}")

                # Label Propagation for Unmatched Nodes
                g_t = propagate_labels(g_t)
                per_image_graphs[idx + 1] = g_t  # Update the graph with propagated labels

            # Segmentation on the last graph
            if len(per_image_graphs) > 0:
                g_last = per_image_graphs[-1]
                x = g_last.x  # Shape: [N, C]
                
                # Reshape x to [1, C, H, W]
                x_image = x.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, C, H, W]
                
                # Apply segmentation head
                logits = segmentation_head(x_image)  # Shape: [1, num_classes, H, W]
                
                # Prepare ground truth mask correctly
                target = masks[b].argmax(dim=0).unsqueeze(0)  # Shape: [1, H, W]
                
                # Compute segmentation loss
                loss_segmentation = criterion(logits, target)
                batch_loss_segmentation += loss_segmentation

        # Combine all losses
        total_loss = batch_loss_mapping + batch_loss_consistency + batch_loss_matching + batch_loss_segmentation

        # Backpropagate total loss
        total_loss.backward()

        # Update parameters
        optimizer.step()

        # Reset gradients for next iteration
        optimizer.zero_grad()

        # Optionally, print losses
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"Batch [{batch_idx+1}/{len(train_loader)}] Loss Mapping: {batch_loss_mapping.item():.4f}")
        print(f"Batch [{batch_idx+1}/{len(train_loader)}] Loss Consistency: {batch_loss_consistency.item():.4f}")
        print(f"Batch [{batch_idx+1}/{len(train_loader)}] Loss Matching: {batch_loss_matching.item():.4f}")
        print(f"Batch [{batch_idx+1}/{len(train_loader)}] Loss Segmentation: {batch_loss_segmentation.item():.4f}")
        print(f"Batch [{batch_idx+1}/{len(train_loader)}] Total Loss: {total_loss.item():.4f}\n")

