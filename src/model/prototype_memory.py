"""Permanent trainable prototype bank and attention usage statistics."""

import torch
from torch import nn


class TrafficPrototype(nn.Module):
    """Store permanent prototypes used by the per-window exchange module."""

    def __init__(self, num_prototypes, prototype_dim):
        nn.Module.__init__(self)
        self.num_prototypes = int(num_prototypes)
        self.prototype_dim = int(prototype_dim)
        self.A = nn.Parameter(torch.randn(self.num_prototypes, self.prototype_dim))
        self.last_attn = None
        self.register_buffer('_usage_counts', torch.zeros(self.num_prototypes))
        self.register_buffer('_attention_sums', torch.zeros(self.num_prototypes))
        self.register_buffer('_total_samples', torch.tensor(0.0))

    def record_attention(self, attention, record_statistics=False):
        """Record actual node-to-prototype retrieval, including dynamic readback."""
        self.last_attn = attention.detach()
        if self.training and record_statistics:
            with torch.no_grad():
                totals = attention.detach().sum(dim=tuple(range(attention.ndim-1)))
                self._usage_counts += totals
                self._attention_sums += totals
                self._total_samples += attention.numel() // attention.size(-1)

    def _prototype_matrix(self):
        return self.A

    def get_prototype_vectors(self):
        return self.A.detach().cpu().numpy()

    def get_prototype_values(self):
        return self.A.detach().cpu().numpy()

    def get_routing_statistics(self):
        if self._total_samples.item() == 0:
            return None
        usage_counts = self._usage_counts.clone()
        usage_ratio = usage_counts / self._total_samples
        attention_mean = self._attention_sums / self._total_samples
        return {
            "usage_counts": usage_counts.cpu().numpy(),
            "usage_ratio": usage_ratio.cpu().numpy(),
            "attention_mean": attention_mean.cpu().numpy(),
        }

    def reset_statistics(self):
        self._usage_counts.zero_()
        self._attention_sums.zero_()
        self._total_samples.zero_()
