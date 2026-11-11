"""Per-window node -> prototype -> node cross attention, without N x N attention."""
import math

from torch import nn


class PrototypeExchange(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = math.sqrt(hidden_dim)
        self.read_query = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.read_key = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.read_value = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.update = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.write_query = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.write_key = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.write_value = nn.Linear(hidden_dim,hidden_dim,bias=False)

    def forward(self, hidden, bank, return_details=False):
        if hidden.ndim != 3 or hidden.size(1) == 0:
            raise ValueError('Prototype exchange requires [batch, nonempty_nodes, hidden]')
        # Softmax over nodes: independently aggregate each window in the batch.
        read = (self.read_query(bank) @ self.read_key(hidden).transpose(-1,-2)/self.scale).softmax(-1)
        pooled = read @ self.read_value(hidden)
        temporary = self.norm(bank.unsqueeze(0)+self.update(pooled))
        # Softmax over prototypes. Temporary states never overwrite the learned bank.
        write = (self.write_query(hidden) @ self.write_key(temporary).transpose(-1,-2)/self.scale).softmax(-1)
        context = write @ self.write_value(temporary)
        return (context,write,read,temporary) if return_details else (context,write)


class PermanentPrototypeRetrieval(nn.Module):
    """One-way node queries retrieve the unchanged permanent bank as K/V."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = math.sqrt(hidden_dim)
        self.write_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.write_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.write_value = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden, bank, return_details=False):
        if hidden.ndim != 3 or hidden.size(1) == 0:
            raise ValueError('Prototype retrieval requires [batch, nonempty_nodes, hidden]')
        weights = (self.write_query(hidden) @ self.write_key(bank).t() / self.scale).softmax(-1)
        context = weights @ self.write_value(bank)
        return (context, weights, None, bank) if return_details else (context, weights)
