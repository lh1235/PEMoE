"""Training-only representative node selection by greedy facility location."""
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.model.node_history import history_statistics, training_history


@dataclass
class RepresentativeSelection:
    node_ids: np.ndarray
    profiles: np.ndarray
    standardized_profiles: np.ndarray
    profile_center: np.ndarray
    profile_scale: np.ndarray
    assignments: np.ndarray
    population_weights: np.ndarray
    error_curve: np.ndarray
    summary: dict


def _distances(left, right):
    return np.maximum((left*left).sum(1)[:, None]
                      + (right*right).sum(1)[None, :] - 2*left@right.T, 0.)


def assign_representatives(profiles, selected_ids):
    """Map each full-graph node to one visible representative, including itself."""
    ids = np.asarray(selected_ids, dtype=np.int64)
    distances = _distances(profiles, profiles[ids])
    positions = distances.argmin(axis=1)
    # Equal/duplicate profiles must not give a selected node zero population mass.
    positions[ids] = np.arange(len(ids))
    weights = np.bincount(positions, minlength=len(ids)).astype(np.int64)
    return ids[positions], weights


def select_representative_nodes(profiles, max_nodes, mandatory_nodes=()):
    """Minimize nearest-representative squared distances at a fixed seed budget.

    Profiles are already imputed raw descriptors. Distances use current-year
    population median/IQR normalization and clipping to [-5,5]. The normalized
    reconstruction error is recorded for diagnostics only. Mandatory nodes
    are never discarded, including when they exceed max_nodes; otherwise the
    seed budget is filled even when reconstruction error has reached zero.
    """
    started = perf_counter()
    raw = np.asarray(profiles, dtype=np.float64)
    if raw.ndim != 2 or not len(raw) or not raw.shape[1] or not np.isfinite(raw).all():
        raise ValueError('Representative profiles must be a finite nonempty [nodes,features] array')
    n = len(raw)
    if int(max_nodes) < 1:
        raise ValueError('Representative selection requires a positive node budget')
    cap = min(int(max_nodes), n)
    selected = list(dict.fromkeys(int(i) for i in mandatory_nodes))
    if any(i < 0 or i >= n for i in selected):
        raise ValueError('Mandatory representative IDs must belong to the current graph')
    mandatory_count = len(selected)
    limit = max(cap, mandatory_count)
    center = np.median(raw, axis=0)
    scale = np.quantile(raw, .75, axis=0)-np.quantile(raw, .25, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    z = np.clip((raw-center)/scale, -5., 5.)
    distances = _distances(z, z)
    np.fill_diagonal(distances, 0.)
    baseline = float(np.square(z-z.mean(axis=0)).sum())
    if not selected:
        # The best single real-node representative initializes the greedy set.
        selected.append(int(distances.sum(axis=0).argmin()))
    nearest = distances[:, selected].min(axis=1)

    def error_ratio():
        return float(nearest.sum()/baseline) if baseline > 1e-12 else 0.

    curve = [[len(selected), error_ratio()]]
    while len(selected) < limit:
        gains = np.maximum(nearest[:, None]-distances, 0.).sum(axis=0)
        gains[selected] = -np.inf
        candidate = int(gains.argmax())
        selected.append(candidate)
        nearest = np.minimum(nearest, distances[:, candidate])
        curve.append([len(selected), error_ratio()])
    ids = np.asarray(selected, dtype=np.int64)
    assignments, weights = assign_representatives(z, ids)
    achieved = error_ratio()
    stop_reason = 'mandatory_exceeds_budget' if mandatory_count > cap else 'node_budget'
    summary = {
        'strategy': 'representative', 'total_nodes': n, 'profile_dim': raw.shape[1],
        'max_seed_nodes': cap, 'mandatory_seed_nodes': mandatory_count, 'selected_seed_nodes': len(ids),
        'profile_reconstruction_error_ratio': achieved,
        'population_center_sse': baseline, 'stop_reason': stop_reason,
        'normalization': 'current training population median/IQR; constant-feature scale=1; clip [-5,5]',
        'distance_matrix_mb': distances.nbytes/1024**2,
        'selection_seconds': perf_counter()-started,
        'weights_applied_to_training': False,
    }
    return RepresentativeSelection(ids, raw, z, center, scale, assignments, weights,
                                   np.asarray(curve, dtype=np.float64), summary)


def sample_training_nodes(inputs, args, mandatory_nodes, max_nodes):
    """Reuse NodeHistory's 28 raw statistics, without modifying its EMA table."""
    started = perf_counter()
    n = int(args.graph_size)
    values, hours = training_history(
        inputs, np.arange(n),
        steps_per_hour=int(getattr(args, 'history_steps_per_hour', 12)),
        start_step=int(getattr(args, 'history_start_step', 0)),
        input_length=int(args.x_len),
    )
    profiles, counts = history_statistics(values, hours)
    # A missing time-of-day bin must not be interpreted as observed zero flow.
    for feature in range(profiles.shape[1]):
        valid = counts[:, feature] > 0
        fallback = np.median(profiles[valid, feature]) if valid.any() else 0.
        profiles[~valid, feature] = fallback
    result = select_representative_nodes(profiles, max_nodes, mandatory_nodes)
    result.summary.update({
        'year': int(args.year), 'history_source': 'unique raw observations reconstructed from train_y only',
        'history_observations': len(values),
        'missing_profile_components_imputed': int((counts == 0).sum()),
        'nodes_without_valid_history': int((counts[:, 24] == 0).sum()),
        'profile_and_selection_seconds': perf_counter()-started,
    })
    return result


def save_selection(result, context_node_ids, directory):
    """Save ordered seed/context mappings and their representative population mass.

    These masses are saved for diagnostics only;
    neither forecast loss nor prototype read attention is changed here.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    context_ids = np.asarray(context_node_ids, dtype=np.int64)
    context_assignments, context_weights = assign_representatives(result.standardized_profiles, context_ids)
    summary = dict(result.summary, context_nodes=len(context_ids),
                   selected_node_ids=result.node_ids.tolist(),
                   seed_population_weights=result.population_weights.tolist(),
                   error_curve=result.error_curve.tolist())
    (directory/'node_selection.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(
        directory/'node_selection.npz', profiles=result.profiles,
        standardized_profiles=result.standardized_profiles,
        profile_center=result.profile_center, profile_scale=result.profile_scale,
        seed_node_ids=result.node_ids, seed_assignments=result.assignments,
        seed_population_weights=result.population_weights, error_curve=result.error_curve,
        context_node_ids=context_ids, context_assignments=context_assignments,
        context_population_weights=context_weights,
    )
    return summary
