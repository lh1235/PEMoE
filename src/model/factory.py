"""Build the representative-sampling PeMoE architecture."""
from .sparse_expert_pemoe import SparseExpertPEMoE


def validate_configuration(args):
    required = {'model_type': 'sparse_expert_pemoe', 'strategy': 'incremental',
                'input_norm': 'zscore',
                'loss': 'mse', 'best_model_criterion': 'validation_full'}
    for key, expected in required.items():
        if getattr(args, key, expected) != expected:
            raise ValueError(f'{key} must be {expected!r}; other variants have been removed')
    for key in ('use_feature_embedding', 'init', 'increase'):
        if getattr(args, key, True) is not True:
            raise ValueError(f'{key}=false is no longer supported')
    if getattr(args, 'node_sampling_strategy', 'representative') != 'representative':
        raise ValueError('node_sampling_strategy must be representative')
    prototype = bool(getattr(args, 'use_prototype', True))
    moe = bool(getattr(args, 'use_moe', True))
    if not prototype and not moe:
        raise ValueError('At least one prediction branch must remain enabled')
    if not prototype and getattr(args, 'use_prototype_exchange', True):
        raise ValueError('Prototype exchange requires the prototype branch')
    if not prototype and any(float(getattr(args, key, 0.)) != 0. for key in
                             ('prototype_compactness_weight', 'prototype_separation_weight')):
        raise ValueError('Prototype regularizers must be zero without prototypes')
    if prototype and not getattr(args, 'use_prototype_gate', True):
        raise ValueError('The remaining prototype branch requires its enhancement gate')
    if int(args.num_prototypes) < 1 or not 1 <= int(args.moe_top_k) <= int(args.num_experts):
        raise ValueError('Invalid prototype count or Top-k expert count')
    for key in ('detect', 'ewc', 'moe_gate_use_anomaly', 'moe_fallback_full'):
        if getattr(args, key, False):
            raise ValueError(f'{key} is no longer supported')
    if getattr(args, 'moe_entropy_threshold', None) is not None:
        raise ValueError('Entropy fallback has been removed')
    if getattr(args, 'moe', None) or getattr(args, 'moe_fallback_top_k', None) is not None:
        raise ValueError('Legacy MoE configuration is no longer supported')
    if not 0 < float(args.node_list_target_ratio) <= 1:
        raise ValueError('node_list_target_ratio must be in (0,1]')


def build_model(args):
    validate_configuration(args)
    return SparseExpertPEMoE(args)
