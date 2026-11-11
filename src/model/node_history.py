"""Training-only node descriptors, updated once per stage and saved with the model."""
import numpy as np
import torch
from torch import nn


def training_history(inputs, node_ids, steps_per_hour=12, start_step=0, input_length=12):
    """Recover unique raw observations from this repo's reverse stride-one targets.

    generate_dataset's earliest target begins at x_len, and its final time point
    is train_end-2. No validation/test data or normalized train_x values are used.
    """
    if steps_per_hour <= 0:
        raise ValueError('history_steps_per_hour must be positive')
    y = np.asarray(inputs['train_y'])
    if y.ndim != 3 or len(y) == 0:
        raise ValueError('Node history requires nonempty [samples,time,nodes] train_y')
    ids = np.asarray(node_ids, dtype=np.int64)
    # Validate overlaps in bounded chunks, without materializing a second full
    # [windows,time,nodes] array when the sampler requests every node.
    for start in range(0, len(y)-1, 128):
        stop = min(start+128, len(y)-1)
        newer = y[start:stop, :-1][:, :, ids]
        older = y[start+1:stop+1, 1:][:, :, ids]
        if not np.allclose(newer, older, equal_nan=True):
            raise ValueError('History requires unshuffled reverse stride-one training windows')
    first = y[-1][:, ids]
    tail = y[-2::-1, -1][:, ids]
    values = np.concatenate([first, tail], axis=0).astype(np.float64, copy=False)
    offset = int(start_step) + int(input_length)
    hours = ((np.arange(len(values)) + offset) // steps_per_hour) % 24
    return values, hours


def history_statistics(values, hours):
    """28 descriptors and per-component valid counts; NaN/Inf never form pairs."""
    nodes = values.shape[1]
    stats, counts = np.zeros((nodes, 28)), np.zeros((nodes, 28))
    for n in range(nodes):
        x = values[:, n]
        valid = np.isfinite(x)
        for hour in range(24):
            group = x[valid & (hours == hour)]
            if group.size:
                stats[n, hour], counts[n, hour] = group.mean(), group.size
        observed = x[valid]
        if observed.size:
            stats[n, 24:26] = observed.mean(), observed.std()
            counts[n, 24:26] = observed.size
        pairs = valid[:-1] & valid[1:]
        left, right = x[:-1][pairs], x[1:][pairs]
        if left.size:
            stats[n, 26] = np.abs(right-left).mean()
            counts[n, 26] = left.size
        if left.size >= 2:
            denom = left.std() * right.std()
            stats[n, 27] = (np.mean((left-left.mean())*(right-right.mean())) / denom
                            if denom > 1e-12 else 0.0)
            counts[n, 27] = left.size
    return stats, counts


class NodeHistory(nn.Module):
    def __init__(self, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(28, embedding_dim), nn.GELU())
        self.register_buffer('profiles', torch.empty(0, 28))
        self.register_buffer('counts', torch.empty(0, 28))
        self.register_buffer('fallback', torch.zeros(28))
        self.register_buffer('center', torch.zeros(28))
        self.register_buffer('scale', torch.ones(28))
        self.register_buffer('last_year', torch.tensor(-1, dtype=torch.long))

    @torch.no_grad()
    def update(self, inputs, node_ids, total_nodes, year, alpha=.3, smoothing=24.,
               steps_per_hour=12, start_step=0, input_length=12):
        if not 0 <= alpha <= 1 or smoothing < 0:
            raise ValueError('History alpha must be in [0,1] and smoothing nonnegative')
        if year == self.last_year.item():
            return False
        if year < self.last_year.item():
            raise ValueError('Cannot update node history backwards in time')
        ids = np.asarray(torch.as_tensor(node_ids).cpu(), dtype=np.int64)
        if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= total_nodes:
            raise ValueError('History update requires unique valid global node IDs')
        if total_nodes < len(self.profiles):
            raise ValueError('History expects stable prefix node IDs on an expanding graph')
        values, hours = training_history(inputs, ids, steps_per_hour, start_step, input_length)
        raw, count = history_statistics(values, hours)
        raw = torch.as_tensor(raw, device=self.center.device, dtype=self.center.dtype)
        count = torch.as_tensor(count, device=self.center.device, dtype=self.center.dtype)
        observed = count > 0
        if self.last_year.item() < 0:
            if not observed.any():
                raise ValueError('Cannot initialize history without finite training observations')
            mean = (raw * count).sum(0) / count.sum(0).clamp_min(1)
            mean[:24] = torch.where(count[:, :24].sum(0) > 0, mean[:24], mean[24])
            self.fallback.copy_(mean)
            # First-stage physical-unit scale, fixed for all subsequent years.
            spread = torch.sqrt(((raw[:, 24]-mean[24]).square() + raw[:, 25].square())
                                .mul(count[:, 24]).sum() / count[:, 24].sum().clamp_min(1)).clamp_min(1e-4)
            self.center[:25].fill_(mean[24])
            self.scale[:27].fill_(spread)
        old_size = len(self.profiles)
        if total_nodes > old_size:
            self.profiles = torch.cat([self.profiles, self.fallback.expand(total_nodes-old_size, -1).clone()])
            self.counts = torch.cat([self.counts, self.counts.new_zeros(total_nodes-old_size, 28)])
        index = torch.as_tensor(ids, device=self.center.device)
        previous, old_counts = self.profiles[index], self.counts[index]
        confidence = count / (count + smoothing).clamp_min(1)
        # New components shrink toward global history; old ones receive an EMA.
        rate = torch.where(old_counts > 0, alpha * confidence, confidence)
        updated = previous + rate * (raw-previous)
        self.profiles[index] = torch.where(observed, updated, previous)
        self.counts[index] = old_counts + count
        self.last_year.fill_(year)
        return True

    def forward(self, hidden, node_ids=None):
        batch, nodes = hidden.shape[:2]
        ids = (torch.arange(nodes, device=hidden.device) if node_ids is None
               else node_ids.to(hidden.device).long().reshape(-1))
        if ids.numel() not in (nodes, batch*nodes):
            raise ValueError('Node IDs must describe one graph or all batched graphs')
        if (ids < 0).any():
            raise ValueError('Node IDs must be nonnegative')
        raw = self.fallback.expand(ids.numel(), -1).clone()
        known = ids < len(self.profiles)
        raw[known] = self.profiles[ids[known]]
        normalized = ((raw-self.center)/self.scale).clamp(-10, 10)
        z = self.embedding(normalized)
        if ids.numel() == nodes:
            z = z.unsqueeze(0).expand(batch, -1, -1)
        return z.reshape(batch, nodes, -1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ('profiles', 'counts'):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, getattr(self, name).new_empty(state_dict[key].shape))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
