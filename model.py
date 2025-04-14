import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d, Embedding
import numpy as np


# ---------------------------
# Transformer Encoder Modules
# ---------------------------
def get_sinusoid_encoding_table(n_position, d_model):
    """Sinusoid position encoding table."""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even indices
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd indices
    return torch.FloatTensor(sinusoid_table)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=512,
        dropout=0.1,
        activation="relu",
        batch_first=True,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src):
        # Self-attention sublayer
        attn_output, attn = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        # Feed-forward sublayer
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src, attn[:, 0, :]


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src):
        output = src
        attn = None
        for mod in self.layers:
            output, attn = mod(output)
        if self.norm is not None:
            output = self.norm(output)
        return output, attn


class DeepTTG(nn.Module):
    def __init__(
        self,
        n_output=1,
        # compound branch parameters
        c_feature=108,
        MLP_dim=96,
        num_gnn_layers=4,
        learn_eps=True,
        dropout=0.1,
        # transformer parameters
        vocab_size=26,
        d_model=120,
        n_heads=4,
        num_transformer_layers=4,
    ):
        super(DeepTTG, self).__init__()
        # ---- Compound Graph Branch (Optimized via jump‐knowledge) ----
        self.num_layers = num_gnn_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                mlp = Sequential(
                    Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim)
                )
            else:
                mlp = Sequential(
                    Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim)
                )
            # Enable learn_eps by passing train_eps=True
            conv = GINConv(mlp, train_eps=learn_eps)
            self.convs.append(conv)
            self.bns.append(BatchNorm1d(MLP_dim))
        # Jump–knowledge aggregation: concatenate outputs of each layer and compress
        self.jk_linear = Linear(MLP_dim * num_gnn_layers, MLP_dim)
        self.compound_fc = Linear(MLP_dim, 120)  # map to 120 dimensions

        # ---- Protein and Pocket Branches (Transformer-based) ----
        self.src_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding.from_pretrained(
            get_sinusoid_encoding_table(vocab_size, d_model), freeze=True
        )
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer, num_layers=num_transformer_layers
        )
        self.protein_fc = Linear(d_model, 120)
        self.pocket_fc = Linear(d_model, 60)

        # ---- Fusion and Final Prediction ----
        self.fc1 = Linear(120 + 120 + 60, 512)
        self.fc2 = Linear(512, 256)
        self.out = Linear(256, n_output)
        self.dropout = Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        # data: a PyG Data object with attributes:
        #   x (node features), edge_index, batch,
        #   protein (LongTensor sequence indices), pocket (LongTensor sequence indices)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ----- Compound Branch: Multi-layer GIN with Jump-Knowledge -----
        layer_outputs = []
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            layer_outputs.append(h)
        # Concatenate features from all layers
        h_cat = torch.cat(layer_outputs, dim=1)
        # Global pooling (sum) over nodes in each graph
        compound_repr = global_add_pool(h_cat, batch)
        # Compress jump-knowledge output
        compound_repr = self.jk_linear(compound_repr)
        compound_repr = F.relu(compound_repr)
        compound_repr = self.compound_fc(compound_repr)
        compound_repr = self.dropout(compound_repr)

        # ----- Protein Branch -----
        # Assume data.protein is a [batch_size, seq_len] LongTensor.
        protein_seq = data.protein
        protein_emb = self.src_emb(protein_seq) + self.pos_emb(protein_seq)
        protein_encoded, _ = self.transformer_encoder(protein_emb)
        protein_repr = protein_encoded[:, 0, :]  # use first token
        protein_repr = self.protein_fc(protein_repr)

        # ----- Pocket Branch -----
        pocket_seq = data.pocket
        pocket_emb = self.src_emb(pocket_seq) + self.pos_emb(pocket_seq)
        pocket_encoded, _ = self.transformer_encoder(pocket_emb)
        pocket_repr = pocket_encoded[:, 0, :]
        pocket_repr = self.pocket_fc(pocket_repr)

        # ----- Fusion and Final Prediction -----
        fusion = torch.cat([compound_repr, protein_repr, pocket_repr], dim=1)
        x = F.relu(self.fc1(fusion))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.out(x)
        return out
