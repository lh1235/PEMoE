"""Prototype exchange with history-conditioned Top-k expert fusion."""
import torch
from .expert_fusion import ReferenceExpertFusion
from .moe import MoELayer
from .time_domain_pemoe import TimeDomainPEMoE
from .node_history import NodeHistory


class SparseExpertPEMoE(TimeDomainPEMoE):
    def __init__(self, args):
        super().__init__(args)
        self.use_moe = bool(getattr(args, 'use_moe', True))
        self.moe_node_history = bool(getattr(args, 'moe_node_history', True)) and self.use_moe
        if self.use_moe:
            self.moe_layer = MoELayer(
                in_channel=1, out_channel=1, kernel_size=int(args.tcn['kernel_size']),
                dilation=int(args.tcn['dilation']), gate_input_dim=self.hidden_dim+(8 if self.moe_node_history else 0),
                num_experts=int(args.num_experts), gate_hidden_dim=int(args.moe_gate_hidden_dim),
                top_k=int(args.moe_top_k))
            if self.moe_node_history:
                self.moe_layer.node_history = NodeHistory()
            self.grn_moe_fusion = ReferenceExpertFusion(self.hidden_dim, self.hidden_dim, args.dropout)

    def _encode(self, x, adj, need_attention=False, node_ids=None, return_graph=False):
        hidden = self._encode_preprototype_hidden(x, adj)
        fused, query, attention = self._fuse_prototype_hidden(hidden, need_attention=True)
        if self.use_moe:
            flat = hidden.reshape(-1, self.hidden_dim)
            gate_input = flat
            if self.moe_node_history:
                personality = self.moe_layer.node_history(hidden, node_ids)
                gate_input = torch.cat([flat, personality.reshape(-1, 8)], dim=-1)
            expert_output = self.moe_layer(flat.unsqueeze(1), gate_input=gate_input).squeeze(1)
            fused = self.grn_moe_fusion(fused.reshape(-1, self.hidden_dim), expert_output).reshape_as(hidden)
        if return_graph:
            return fused, adj
        return (fused, query, attention) if need_attention else fused

    def get_expert_statistics(self):
        return self.moe_layer.get_expert_statistics() if self.use_moe else None

    def reset_expert_statistics(self):
        if self.use_moe:
            self.moe_layer.reset_statistics()
