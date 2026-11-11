"""Time-domain graph encoder with gated prototype-bank retrieval."""
import torch
from torch import nn
import torch.nn.functional as F

from .gcn_conv import BatchGCNConv
from .prototype_memory import TrafficPrototype
from .prototype_exchange import PrototypeExchange, PermanentPrototypeRetrieval


class TimeDomainPEMoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_len = int(args.gcn['in_channel'])
        self.out_len = int(args.y_len)
        self.hidden_dim = int(args.gcn['hidden_channel'])
        self.prototype_dim = self.hidden_dim
        out_channel = int(args.gcn['out_channel'])
        if out_channel != self.in_len:
            raise ValueError('gcn out_channel must equal in_channel for the input residual')
        self.dropout = float(args.dropout)
        self.input_norm = str(getattr(args, 'input_norm', 'zscore')).strip().lower()
        if self.input_norm != 'zscore':
            raise ValueError('Only zscore input normalization is supported')
        self.register_buffer('target_mean', torch.tensor(float(getattr(args, 'target_mean', 0.0))))
        self.register_buffer('target_std', torch.tensor(max(float(getattr(args, 'target_std', 1.0)), 1e-5)))

        # Same time-domain graph projection as the original paper's code.
        self.gcn1 = BatchGCNConv(self.in_len, self.hidden_dim, bias=True, gcn=False)
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.LayerNorm(self.hidden_dim))
        self.prototype_compactness_loss = None
        self.last_prototype_gate = None
        self.use_prototype = bool(getattr(args, 'use_prototype', True))
        if self.use_prototype:
            self.prototype = TrafficPrototype(int(args.num_prototypes), self.prototype_dim)
            self.prototype_gate = nn.Linear(2*self.hidden_dim, 1)
            self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.gcn2 = BatchGCNConv(self.hidden_dim, out_channel, bias=True, gcn=False)
        self.fc = nn.Linear(out_channel, self.out_len)
        self.activation = nn.GELU()
        if self.use_prototype:
            with torch.random.fork_rng(devices=[]):
                exchange = PrototypeExchange if getattr(args, 'use_prototype_exchange', True) else PermanentPrototypeRetrieval
                self.prototype_exchange = exchange(self.hidden_dim)

    def _encode_preprototype_hidden(self, x, adj):
        hidden = self._encode_gcn_hidden(x, adj)
        return self._embed_hidden(hidden)


    def _encode_gcn_hidden(self, x, adj, return_support=False, node_ids=None):
        hidden = F.relu(self.gcn1(x, adj))
        return (hidden, adj) if return_support else hidden

    def _embed_hidden(self, hidden):
        return self.feature_embedding(hidden)

    def _prototype_mix(self, hidden, return_attn=False):
        context, attention = self.prototype_exchange(hidden, self.prototype.A)
        self.prototype.record_attention(attention, record_statistics=self.training)
        return (context, hidden, attention) if return_attn else (context, hidden)

    def _fuse_prototype_hidden(self, hidden, need_attention=False, node_ids=None):
        if not self.use_prototype:
            return (hidden, hidden, None) if need_attention else hidden
        context, query, attention = self._prototype_mix(hidden, return_attn=True)
        gate = self.prototype_gate(torch.cat([hidden, context], dim=-1)).sigmoid()
        self.last_prototype_gate = gate.detach()
        if float(getattr(self.args, 'prototype_compactness_weight', 0.0)) > 0:
            distance = torch.cdist(query, self.prototype.A).square()
            compact_attention = torch.softmax(query @ self.prototype.A.t(), dim=-1)
            self.prototype_compactness_loss = (compact_attention*distance).sum(-1).mean()
        fused = self.fusion_norm(hidden + gate*context)
        return (fused, query, attention) if need_attention else fused

    def forward(self, data, adj):
        self.prototype_compactness_loss = None
        x = data.x.reshape(-1, adj.shape[0], self.in_len)
        hidden, prediction_adj = self._encode(x, adj, node_ids=getattr(data, 'node_ids', None), return_graph=True)
        prediction = self.gcn2(hidden, prediction_adj) + x
        prediction = F.dropout(prediction, p=self.dropout, training=self.training)
        prediction = self.fc(self.activation(prediction))
        return (prediction*self.target_std+self.target_mean).reshape(-1, self.out_len)


    def get_preprototype_features(self, data, adj):
        x = data.x.reshape(-1, adj.shape[0], self.in_len)
        hidden = self._encode_preprototype_hidden(x, adj)
        return hidden.reshape(-1, self.prototype_dim)

    def get_prototype_kmeans_features(self, data, adj):
        return self.get_preprototype_features(data, adj)

    def get_prototype_attention(self, data, adj):
        if not self.use_prototype:
            return None
        query = self.get_preprototype_features(data, adj)
        _, attention = self.prototype_exchange(
            query.reshape(-1, adj.size(0), self.hidden_dim), self.prototype.A)
        return attention.reshape(-1, self.prototype.num_prototypes)

    def clear_cached_tensors(self):
        self.prototype_compactness_loss = None
        if self.use_prototype:
            self.prototype.last_attn = None
