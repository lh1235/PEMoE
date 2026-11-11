# -*- coding: utf-8 -*-
import gc
import json
import numpy as np
import math
import os.path as osp
import networkx as nx

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

from utils import common_tools as ct
from utils.data_convert import generate_samples
from src.trafficDataset import TrafficDataset
from src.config import ExperimentConfig
from src.trainer import TrafficTrainer
from src.model.factory import validate_configuration
from src.node_sampling import sample_training_nodes, save_selection

result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}, "time_info":{}}
pin_memory = True
n_work = 0


def load_best_model(args):
    return TrafficTrainer(args, result, pin_memory=pin_memory, num_workers=n_work).load_best_model()


def get_year_data_cache(args, year):
    if getattr(args, 'input_norm', 'zscore') != 'zscore':
        raise ValueError('Only zscore caches are supported')
    suffix = '_30day_zscore'
    stem = osp.join(args.save_data_path, f"{year}{suffix}")
    return stem, stem + ".npz"


def build_expansion_edge_index(physical_adj_np, args):
    """Build physical edge_index used for selecting the incremental k-hop subgraph."""
    physical_edges = list(nx.from_numpy_array(physical_adj_np).edges)
    edge_parts = []
    if physical_edges:
        edges = torch.LongTensor(np.array(physical_edges).T)
        # NetworkX emits each undirected edge once; PyG needs both directions
        # for a complete physical k-hop neighbourhood.
        edge_parts.extend([edges, edges.flip(0)])

    if not edge_parts:
        return torch.empty((2, 0), dtype=torch.long), set()

    edge_index = torch.cat(edge_parts, dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    graph_node_from_edge = set(edge_index.reshape(-1).tolist())
    return edge_index, graph_node_from_edge


def train(inputs, args):
    return TrafficTrainer(args, result, pin_memory=pin_memory, num_workers=n_work).fit(inputs)


def test_model(model, args, testset, pin_memory):
    return TrafficTrainer(args, result, pin_memory=pin_memory, num_workers=n_work).test_model(model, testset)


def _mean_or_nan(values):
    return sum(values) / len(values) if values else float("nan")


def _format_summary_line(label, metric_name, values, avg_label, avg_value):
    values_text = " ".join(f"{v:>6.2f}" for v in values) if values else ""
    avg_text = f"{avg_value:>6.2f}" if math.isfinite(avg_value) else f"{'nan':>6}"
    return f"{label:>10} {metric_name:<5} {values_text}  {avg_label:<8} {avg_text}"


def main(args):
    validate_configuration(args)
    logger = ExperimentConfig.init_logger(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):
        if year > args.begin_year:
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"[*] Memory cleaned before year {year}")
        # ============================================================

        # Load Data
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year

        cache_stem, cache_npz = get_year_data_cache(args, year)
        if bool(args.data_process):
            raw_data = np.load(osp.join(args.raw_data_path, str(year) + ".npz"))["x"]
            inputs = generate_samples(
                31,
                cache_stem,
                raw_data,
                graph,
                val_test_mix=False,
                input_norm=args.input_norm,
            )
        else:
            if not osp.exists(cache_npz):
                raise FileNotFoundError(
                    f"Preprocessed data not found at {cache_npz}. "
                    "Set data_process=1 to regenerate it."
                )
            inputs = np.load(cache_npz, allow_pickle=True)

        args.logger.info("[*] Year {} load from {}".format(args.year, cache_npz))

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        adj_tensor = torch.from_numpy(adj).to(torch.float).to(args.device)
        vars(args)["adj_physical"] = adj_tensor
        vars(args)["adj"] = adj_tensor

        if year == args.begin_year and args.load_first_year:
            # Skip the first year, model has been trained and retrain is not needed
            model, _ = load_best_model(args)
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            test_model(model, args, test_loader, pin_memory=True)
            continue

        if year > args.begin_year:
            # Load the best model
            model, _ = load_best_model(args)
            physical_adj_np = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
            cur_graph, graph_node_from_edge = build_expansion_edge_index(physical_adj_np, args)
            graph_node_from_edge = set(range(args.graph_size))
            node_list = list()
            representative_selection = None

            # New nodes are mandatory.
            cur_node_size = physical_adj_np.shape[0]
            pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
            increase_node_list = list(range(pre_node_size, cur_node_size))
            node_list.extend(increase_node_list)
            args.logger.info("[*] increase selected nodes {}".format(len(increase_node_list)))

            node_list = [
                node
                for node in dict.fromkeys(int(node) for node in node_list)
                if node in graph_node_from_edge
            ]
            target_ratio = float(getattr(args, "node_list_target_ratio", 0.2))
            target_num = int(math.ceil(target_ratio * args.graph_size))
            representative_selection = sample_training_nodes(inputs, args, node_list, target_num)
            node_list = representative_selection.node_ids.tolist()
            summary = representative_selection.summary
            logger.info(
                '[*] Representative seeds | selected=%s cap=%s mandatory=%s profile_error=%.6f '
                'stop=%s preprocessing_and_selection=%.3fs matrix=%.2fMB',
                len(node_list), target_num, summary['mandatory_seed_nodes'],
                summary['profile_reconstruction_error_ratio'], summary['stop_reason'],
                summary['profile_and_selection_seconds'], summary['distance_matrix_mb'])

            # Preserve the selection order so diagnostics and seed mapping agree.
            node_list = [node for node in dict.fromkeys(node_list) if node in graph_node_from_edge]


            if len(node_list) != 0 :
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True, num_nodes=args.graph_size)
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            else:
                vars(args)["subgraph"] = torch.empty(0, dtype=torch.long)
                vars(args)["mapping"] = torch.empty(0, dtype=torch.long)
            logger.info("number of selected seed nodes:{}, nodes after {} hop:{}, total nodes this year {}".format\
                        (len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)
            if representative_selection is not None:
                save_selection(representative_selection, args.subgraph.cpu().numpy(),
                               osp.join(args.path, str(year)))
                logger.info('[*] Saved node_selection.json/.npz; population weights are diagnostics only')


        # Skip the year when no nodes needed to be trained incrementally
        if year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)
            ct.mkdirs(osp.join(args.model_path, args.logname+args.time, str(args.year)))
            torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".pkl"))
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            test_model(model, args, test_loader, pin_memory=True)
            logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue


        if args.train:
            train(inputs, args)
        else:
            if args.auto_test:
                model, _ = load_best_model(args)
                test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
                test_model(model, args, test_loader, pin_memory=True)


    years = list(range(args.begin_year, args.end_year + 1))

    for horizon in [3, 6, 12]:
        for metric_name in ["mae", "rmse", "mape"]:
            values = [
                result[horizon][metric_name][year]
                for year in years
                if horizon in result and metric_name in result[horizon] and year in result[horizon][metric_name]
            ]
            if not values:
                continue
            avg_value = _mean_or_nan(values)
            logger.info(_format_summary_line(str(horizon), metric_name, values, f"avg_{metric_name}", avg_value))

    for metric_name in ["mae", "rmse", "mape"]:
        per_year_means = []
        for year in years:
            year_values = [
                result[h][metric_name][year]
                for h in [3, 6, 12]
                if h in result and metric_name in result[h] and year in result[h][metric_name]
            ]
            if year_values:
                per_year_means.append(_mean_or_nan(year_values))
        if per_year_means:
            logger.info(
                _format_summary_line(
                    f"avg{metric_name}",
                    metric_name,
                    per_year_means,
                    f"avg_{metric_name}",
                    _mean_or_nan(per_year_means),
                )
            )

    for year in range(args.begin_year, args.end_year+1):
        if year in result["time_info"]:
            gpu_memory = result["time_info"][year].get("gpu_memory", {})
            logger.info(
                "year %4d  total_time %8.2fs  average_time %7.2fs  epoch %3d  max_allocated_mb %8s  max_reserved_mb %8s",
                year,
                result["time_info"][year]["total_time"],
                result["time_info"][year]["average_time"],
                result["time_info"][year]["epoch_num"],
                "{:.2f}".format(gpu_memory["max_allocated_mb"]) if "max_allocated_mb" in gpu_memory else "NA",
                "{:.2f}".format(gpu_memory["max_reserved_mb"]) if "max_reserved_mb" in gpu_memory else "NA",
            )


    with open(osp.join(args.path, 'metrics.json'), 'w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, default=float)


if __name__ == "__main__":
    args = ExperimentConfig.from_cli()
    main(args)
