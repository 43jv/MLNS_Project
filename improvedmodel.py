import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# Enhanced transformer constants
d_model = 128  # Increased from 120
dim_feedforward = 768  # Increased from 512
n_heads = 8  # Increased from 4
vocab_size = 26
n_layers = 6  # Increased from 4
dropout_rate = 0.2  # Increased slightly for better regularization


class TransformerEncoderWithPE(nn.Module):
    """Enhanced transformer encoder with improved positional encoding and layer norm"""

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout=0.1,
        max_len=1000,
        activation="gelu",
    ):
        super(TransformerEncoderWithPE, self).__init__()

        # Positional encoding
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len)

        # Pre-norm architecture (more stable training)
        encoder_layer = TransformerEncoderLayerPreNorm(
            d_model, nhead, dim_feedforward, dropout, activation
        )

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        # Apply positional encoding
        output = self.pos_embedding(src)

        # Store attention weights for visualization
        attention_weights = []

        # Process through transformer layers
        for layer in self.layers:
            output, attn = layer(output, src_mask)
            attention_weights.append(attn)

        output = self.norm(output)

        # Return the output and the attention from the last layer
        return output, attention_weights[-1]


# Integrates the positional encoding as a non-trainable part of the model (by registering it as a buffer).
# These encodings are constant and reliable every time the model processes a sequence.
class PositionalEncoding(nn.Module):
    """Improved sinusoidal positional encoding"""

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix handles longer sequences
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter but part of the module)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        # Apply dropout to the positional encoding to prevent overfitting
        return self.dropout(x)


class TransformerEncoderLayerPreNorm(nn.Module):
    """Pre-normalization transformer encoder layer (more stable training)"""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
        batch_first=True,
    ):
        super(TransformerEncoderLayerPreNorm, self).__init__()

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # Implementation of feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Layer norms and dropouts
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None):
        # Pre-norm architecture: apply norm before attention
        src2 = self.norm1(src)
        src2, attn = self.self_attn(src2, src2, src2, attn_mask=src_mask)
        src = src + self.dropout1(src2)

        # Pre-norm for feedforward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src, attn


class MultiHeadSelfAttention(nn.Module):
    """Custom implementation of multi-head self-attention for flexibility"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        # Linear projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Linear projections and reshape to (batch, heads, seq, head_dim)
        q = (
            self.query(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Reshape and project
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.d_model)
        )
        output = self.out_proj(context)

        return output, attn_weights


class EnhancedProteinEncoder(nn.Module):
    """Specialized encoder for protein sequences with additional features"""

    def __init__(
        self, vocab_size, d_model, n_heads, n_layers, dim_feedforward, dropout=0.1
    ):
        super(EnhancedProteinEncoder, self).__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder with pre-norm
        self.transformer = TransformerEncoderWithPE(
            d_model, n_heads, n_layers, dim_feedforward, dropout
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Get embeddings
        x = self.token_embedding(x)

        # Pass through transformer
        x, attention = self.transformer(x)

        # Use first token ([CLS] equivalent) as sequence representation
        seq_repr = x[:, 0, :]

        # Project output
        output = self.output_projection(seq_repr)

        return output, attention, x


class SE3CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(SE3CrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query, key, value):
        # query, key, and value: [batch_size, seq_len, d_model]
        attn_output, attn_weights = self.cross_attn(query, key, value)
        return attn_output, attn_weights


# ImprovedDeepTTG Model
class ImprovedDeepTTG(nn.Module):
    def __init__(
        self,
        n_output=1,
        MLP_dim=96,
        dropout=0.2,  # dropout_rate increased to 0.2
        c_feature=108,
        vocab_size=26,
        d_model=128,  # Increased from 120 to 128
        n_heads=8,  # Increased number of heads
        n_layers=6,  # Increased number of transformer layers
        dim_feedforward=768,  # Increased feedforward dimension
    ):
        super(ImprovedDeepTTG, self).__init__()

        # Enhanced protein encoder
        self.protein_encoder = EnhancedProteinEncoder(
            vocab_size, d_model, n_heads, n_layers, dim_feedforward, dropout
        )

        # Pocket encoder (sharing architecture with protein encoder)
        self.pocket_encoder = EnhancedProteinEncoder(
            vocab_size, d_model, n_heads, n_layers, dim_feedforward, dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.n_output = n_output

        # GIN model for extracting compound features (kept same as original)
        nn1 = Sequential(Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(MLP_dim)
        nn2 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(MLP_dim)
        nn3 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(MLP_dim)
        nn4 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(MLP_dim)

        self.fc1_c = Linear(MLP_dim, d_model)
        self.poc_fc = Linear(
            d_model, d_model // 2
        )  # Pocket branch remains projecting to half dimension

        # Initialize Cross-Attention between Protein and Pocket features
        self.se3_cross_attention = SE3CrossAttention(
            d_model, n_heads=4, dropout=dropout
        )

        # Update fusion layer to accept concatenated features:
        # protein global feature: d_model (128)
        # pocket global feature: d_model//2 (64)
        # ligand (compound) feature: d_model (128)
        # cross-attended feature: d_model (128)
        fusion_input_dim = (
            d_model + (d_model // 2) + d_model + d_model
        )  # 128 + 64 + 128 + 128 = 448 (if d_model==128)
        self.fusion_layer = nn.Linear(fusion_input_dim, dim_feedforward)
        self.layer_norm1 = nn.LayerNorm(dim_feedforward)

        self.fc2 = nn.Linear(dim_feedforward, dim_feedforward // 2)
        self.layer_norm2 = nn.LayerNorm(dim_feedforward // 2)

        self.out = nn.Linear(dim_feedforward // 2, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.protein
        pocket = data.pocket

        # Compound Branch: Feature Extraction
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = global_add_pool(x, batch)
        ligand_feat = F.relu(self.fc1_c(x))
        ligand_feat = self.dropout(ligand_feat)

        # Protein & Pocket Branches: Transformer Encoders
        # Each encoder returns a global feature, attention weights, and the full sequence representation.
        pro_feat, pro_attn, protein_seq = self.protein_encoder(target)
        poc_feat, poc_attn, pocket_seq = self.pocket_encoder(pocket)
        poc_feat = self.poc_fc(poc_feat)

        # Cross-Attention between Protein and Pocket sequences
        # Here, we use the full sequence outputs (protein_seq and pocket_seq) for cross-attention.
        cross_output, cross_attn = self.se3_cross_attention(
            protein_seq, pocket_seq, pocket_seq
        )
        # Use the first token as the aggregated cross-attended feature
        cross_feature = cross_output[:, 0, :]

        # Concatenate protein, pocket, compound, and cross-attended features
        x = torch.cat([pro_feat, poc_feat, ligand_feat, cross_feature], dim=1)
        fusion = self.fusion_layer(x)
        fusion = self.gelu(fusion)
        fusion = self.dropout(fusion)
        fusion = self.layer_norm1(fusion)

        # Second fusion layer with skip connection and normalization
        x = self.fc2(fusion)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer_norm2(x)

        # Final output layer
        out = self.out(x)
        return out, pro_attn, poc_attn


# Utility function to create a subset of the dataset for quick testing
def create_dataset_subset(dataset, fraction=0.1, min_samples=20):
    """Create a subset of the dataset for quick testing"""
    import random

    total_samples = len(dataset)
    subset_size = max(min_samples, int(total_samples * fraction))

    # Randomly select indices
    indices = random.sample(range(total_samples), subset_size)

    return torch.utils.data.Subset(dataset, indices)
