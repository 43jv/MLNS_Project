import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from torch_geometric.nn import GINConv, global_add_pool


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayerPreNorm(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.2,
        activation="gelu",
        layer_norm_eps=1e-5,
        batch_first=True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None):
        # Pre-norm + self-attention
        x = self.norm1(src)
        x2, attn = self.self_attn(x, x, x, attn_mask=src_mask)
        src = src + self.dropout1(x2)

        # Pre-norm + feedforward
        x = self.norm2(src)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout2(x2)
        return src, attn


class TransformerEncoderWithPE(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout=0.2,
        max_len=1000,
        activation="gelu",
    ):
        super().__init__()
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len)
        layer = TransformerEncoderLayerPreNorm(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.layers = _get_clones(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        # src: (batch, seq_len, d_model)
        x = self.pos_embedding(src)
        attn_w = None
        for layer in self.layers:
            x, attn_w = layer(x, src_mask)
        x = self.norm(x)
        return x, attn_w


class SE3CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, q, k, v):
        out, w = self.cross_attn(q, k, v)
        return out, w


class EnhancedProteinEncoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_heads, n_layers, dim_feedforward, dropout=0.2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.transformer = TransformerEncoderWithPE(
            d_model, n_heads, n_layers, dim_feedforward, dropout
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len) integer tokens
        x = self.embedding(x)
        x, attn = self.transformer(x)
        cls = x[:, 0, :]
        return self.proj(cls), attn, x


class ImprovedDeepTTG(nn.Module):
    def __init__(
        self,
        n_output=1,
        MLP_dim=96,
        dropout=0.2,
        c_feature=108,
        vocab_size=26,
        d_model=128,
        n_heads=8,
        n_layers=6,
        dim_feedforward=768,
        struct_dim=128,
    ):
        super().__init__()

        # Protein & pocket encoders
        self.prot_enc = EnhancedProteinEncoder(
            vocab_size, d_model, n_heads, n_layers, dim_feedforward, dropout
        )
        self.pock_enc = EnhancedProteinEncoder(
            vocab_size, d_model, n_heads, n_layers, dim_feedforward, dropout
        )

        # Ligand GIN branch
        # first GINConv: input dim = c_feature
        gin1 = Sequential(Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv1, self.bn1 = GINConv(gin1), nn.BatchNorm1d(MLP_dim)
        # subsequent GINConvs: input = output = MLP_dim
        gin2 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv2, self.bn2 = GINConv(gin2), nn.BatchNorm1d(MLP_dim)
        self.conv3, self.bn3 = GINConv(gin2), nn.BatchNorm1d(MLP_dim)
        self.conv4, self.bn4 = GINConv(gin2), nn.BatchNorm1d(MLP_dim)

        self.fc1_c = Linear(MLP_dim, d_model)
        self.poc_fc = Linear(d_model, d_model // 2)

        # Cross-attention
        self.cross_attn = SE3CrossAttention(d_model, n_heads=4, dropout=dropout)

        # Structure GIN branch
        struct_nn = Sequential(Linear(3, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.struct_conv1 = GINConv(struct_nn)
        self.struct_bn1 = nn.BatchNorm1d(MLP_dim)
        self.struct_fc = Linear(MLP_dim, struct_dim)

        # Fusion
        fuse_dim = (
            d_model  # protein
            + (d_model // 2)  # pocket
            + d_model  # ligand
            + d_model  # cross-attn
            + struct_dim  # structure (or zero‑pad)
        )
        self.fusion1 = Linear(fuse_dim, dim_feedforward)
        self.norm_fuse1 = nn.LayerNorm(dim_feedforward)
        self.fusion2 = Linear(dim_feedforward, dim_feedforward // 2)
        self.norm_fuse2 = nn.LayerNorm(dim_feedforward // 2)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim_feedforward // 2, n_output)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        prot_seq, pock_seq = data.protein, data.pocket

        # — Ligand branch —
        x = F.relu(self.conv1(x, ei))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, ei))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, ei))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, ei))
        x = self.bn4(x)
        x = global_add_pool(x, batch)
        ligand_feat = F.relu(self.fc1_c(x))
        ligand_feat = self.dropout(ligand_feat)

        # — Protein & pocket encoders —
        pro_feat, pro_attn, pro_seq_enc = self.prot_enc(prot_seq)
        poc_feat, poc_attn, poc_seq_enc = self.pock_enc(pock_seq)
        poc_feat = self.poc_fc(poc_feat)

        # — Cross‑attention —
        cross_out, _ = self.cross_attn(pro_seq_enc, poc_seq_enc, poc_seq_enc)
        cross_feat = cross_out[:, 0, :]

        # — Structure branch —
        if hasattr(data, "prot_str_x") and hasattr(data, "prot_str_edge_index"):
            coords = data.prot_str_x
            eidx = data.prot_str_edge_index
            s = F.relu(self.struct_conv1(coords, eidx))
            s = self.struct_bn1(s)
            s = global_add_pool(s, batch)
            struct_feat = F.relu(self.struct_fc(s))
        else:
            batch_size = ligand_feat.size(0)
            struct_feat = ligand_feat.new_zeros(batch_size, self.struct_fc.out_features)

        # — Fuse all modalities —
        fuse = torch.cat(
            [pro_feat, poc_feat, ligand_feat, cross_feat, struct_feat], dim=1
        )
        f = self.fusion1(fuse)
        f = F.gelu(f)
        f = self.dropout(f)
        f = self.norm_fuse1(f)
        f2 = self.fusion2(f)
        f2 = F.gelu(f2)
        f2 = self.dropout(f2)
        f2 = self.norm_fuse2(f2)

        out = self.out(f2)
        return out, pro_attn, poc_attn


def create_dataset_subset(dataset, fraction=0.1, min_samples=20):
    import random

    total = len(dataset)
    size = max(min_samples, int(total * fraction))
    idx = random.sample(range(total), size)
    return torch.utils.data.Subset(dataset, idx)
