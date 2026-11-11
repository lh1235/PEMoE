"""Gated residual fusion of prototype features and expert context."""
from torch import nn


class ReferenceExpertFusion(nn.Module):
    """Combine the prototype branch with expert context."""
    def __init__(self, hidden_dim, context_dim=None, dropout=0.0):
        super(ReferenceExpertFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim if context_dim is not None else hidden_dim

        self.W2 = nn.Linear(hidden_dim, hidden_dim)

        self.W3 = nn.Linear(self.context_dim, hidden_dim)

        self.W1 = nn.Linear(hidden_dim, hidden_dim)

        self.elu = nn.ELU()

        self.sigmoid = nn.Sigmoid()

        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x, context):
        """Combine the prototype branch with expert context."""
        eta2 = self.elu(self.W2(x) + self.W3(context))

        eta1 = self.W1(eta2)
        eta1 = self.dropout(eta1)

        gate = self.sigmoid(eta1)
        delta = gate * eta1

        y = self.layer_norm(x + delta)

        return y
