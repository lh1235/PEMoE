import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """Independent feature-axis convolutions with Top-k mixture weights."""
    def __init__(self, num_experts, in_channel, out_channel, kernel_size, dilation,
                 gate_hidden_dim=None, gate_input_dim=None, top_k=None):
        """Independent feature-axis convolutions with Top-k mixture weights."""
        super(MoELayer, self).__init__()

        self.num_experts = num_experts
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.top_k = top_k if top_k is not None else num_experts
        if not 1 <= self.top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        padding = int((kernel_size - 1) * dilation / 2)

        self.expert_weight = nn.Parameter(
            torch.randn(num_experts, out_channel, in_channel, kernel_size)
        )
        self.expert_bias = nn.Parameter(torch.zeros(num_experts, out_channel))

        for i in range(num_experts):
            nn.init.kaiming_normal_(self.expert_weight[i], mode='fan_out', nonlinearity='relu')

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding

        gate_in = gate_input_dim if gate_input_dim else in_channel
        gate_hidden = gate_hidden_dim if gate_hidden_dim else in_channel
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_experts)
        )

        self.register_buffer('_expert_usage_counts', torch.zeros(num_experts))
        self.register_buffer('_expert_gate_probs', torch.zeros(num_experts))
        self.register_buffer('_total_samples', torch.tensor(0.0))

    def routing_logits(self, gate_input):
        return self.gate(gate_input)

    def forward(self, x, gate_input):
        """Independent feature-axis convolutions with Top-k mixture weights."""
        B, C, L = x.shape
        E = self.num_experts
        C_out = self.out_channel

        gate_logits = self.routing_logits(gate_input)
        topk_logits, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        gate_weights = torch.zeros_like(gate_logits)
        gate_weights.scatter_(1, topk_indices, F.softmax(topk_logits, dim=-1))

        x_grouped = x.unsqueeze(1).expand(B, E, C, L)  # [B, E, C, L]
        x_grouped = x_grouped.reshape(B, E * C, L)      # [B, E*C, L]

        grouped_weight = self.expert_weight.reshape(E * C_out, C, self.kernel_size)

        grouped_bias = self.expert_bias.reshape(E * C_out) if self.expert_bias is not None else None

        expert_outputs_grouped = F.conv1d(
            x_grouped,
            grouped_weight,
            bias=grouped_bias,
            padding=self.padding,
            dilation=self.dilation,
            groups=E
        )  # [B, E*C_out, L]

        expert_outputs_grouped = F.gelu(expert_outputs_grouped)

        expert_outputs = expert_outputs_grouped.reshape(B, E, C_out, L)

        # gate_weights: [B, E]
        # expert_outputs: [B, E, C_out, L]
        # output: [B, C_out, L]

        gate_weights_expanded = gate_weights.unsqueeze(-1).unsqueeze(-1)  # [B, E, 1, 1]
        output = (expert_outputs * gate_weights_expanded).sum(dim=1)  # [B, C_out, L]

        if self.training:
            with torch.no_grad():
                counts = gate_weights.sum(dim=0)
                self._expert_usage_counts += counts
                self._expert_gate_probs += counts  # Retain checkpoint/statistics compatibility.
                self._total_samples += B

        return output

    def get_expert_statistics(self):
        """Independent feature-axis convolutions with Top-k mixture weights."""
        if self._total_samples == 0:
            return None

        usage_counts = self._expert_usage_counts.clone()
        usage_ratio = usage_counts / self._total_samples
        mean_gate_probs = self._expert_gate_probs / self._total_samples

        return {
            'usage_counts': usage_counts.cpu().numpy(),
            'usage_ratio': usage_ratio.cpu().numpy(),
            'mean_gate_probs': mean_gate_probs.cpu().numpy(),
            'num_experts': self.num_experts,
            'top_k': self.top_k,
        }

    def reset_statistics(self):
        """Independent feature-axis convolutions with Top-k mixture weights."""
        self._expert_usage_counts.zero_()
        self._expert_gate_probs.zero_()
        self._total_samples.zero_()
