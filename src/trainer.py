# -*- coding: utf-8 -*-
import os
import os.path as osp
from dataclasses import dataclass
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as func
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from src.model.factory import build_model
from src.trafficDataset import TrafficDataset
from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np


@dataclass
class TrainingSummary:
    best_model_path: str
    best_loss: float
    total_time: float
    average_epoch_time: float
    epoch_num: int


class TrafficTrainer:
    """Owns model construction, training loss, validation loss, testing, and checkpointing."""

    def __init__(self, args, result_store, pin_memory=True, num_workers=0):
        self.args = args
        self.result = result_store
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    def load_best_model(self):
        args = self.args
        if (args.load_first_year and args.year <= args.begin_year + 1) or args.train == 0:
            load_path = args.first_year_model_path
            best_loss = osp.splitext(osp.basename(load_path))[0]
        else:
            prev_year_dir = osp.join(args.model_path, args.logname + args.time, str(args.year - 1))
            losses = []
            for filename in os.listdir(prev_year_dir):
                if not filename.endswith(".pkl"):
                    continue
                loss_str = osp.splitext(filename)[0]
                try:
                    losses.append((float(loss_str), loss_str))
                except ValueError:
                    continue
            if not losses:
                raise FileNotFoundError(f"No valid .pkl model found in {prev_year_dir}")
            losses.sort(key=lambda x: x[0])
            best_loss = losses[0][1]
            load_path = osp.join(prev_year_dir, best_loss + ".pkl")

        args.logger.info("[*] load from {}".format(load_path))
        state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]
        model = build_model(args)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError('Checkpoint does not match this architecture; use its matching config') from error
        return model.to(args.device), best_loss

    def _uses_cuda(self):
        device = getattr(self.args, "device", None)
        return torch.cuda.is_available() and isinstance(device, torch.device) and device.type == "cuda"

    def _reset_gpu_memory_stats(self):
        if not self._uses_cuda():
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.args.device)

    def _get_gpu_memory_stats(self):
        if not self._uses_cuda():
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_allocated_mb": 0.0,
                "max_reserved_mb": 0.0,
            }
        torch.cuda.synchronize(self.args.device)
        mb = 1024 ** 2
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.args.device) / mb,
            "reserved_mb": torch.cuda.memory_reserved(self.args.device) / mb,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.args.device) / mb,
            "max_reserved_mb": torch.cuda.max_memory_reserved(self.args.device) / mb,
        }

    def fit(self, inputs):
        args = self.args
        path = osp.join(args.path, str(args.year))
        ct.mkdirs(path)
        self._reset_gpu_memory_stats()

        lossfunc = self._build_loss()
        train_loader, val_loader, val_loader_full, test_loader = self._build_loaders(inputs)
        # Calibrate the prediction head from training
        # labels only. Subsequent checkpoints retain their saved calibration.
        args.target_mean = float(train_loader.dataset.y.mean())
        args.target_std = max(float(train_loader.dataset.y.std()), 1e-5)
        args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

        model, gnn_model = self._build_model(inputs, lossfunc)
        ids = (args.subgraph if args.year > args.begin_year
               else torch.arange(args.adj.shape[0]))
        history = getattr(getattr(gnn_model, 'moe_layer', None), 'node_history', None)
        if history is not None:
            history.update(inputs, ids, args.adj.shape[0], int(args.year),
                           alpha=float(getattr(args, 'history_update_alpha', .3)),
                           smoothing=float(getattr(args, 'history_smoothing', 24.)),
                           steps_per_hour=int(getattr(args, 'history_steps_per_hour', 12)),
                           start_step=int(getattr(args, 'history_start_step', 0)),
                           input_length=gnn_model.in_len)
            args.logger.info('[*] Node history: year=%s updated_nodes=%s total_nodes=%s; training split only',
                             args.year, len(ids), len(history.profiles))
        args.logger.info("[*] Target calibration mean=%.6f std=%.6f",
                         gnn_model.target_mean.item(), gnn_model.target_std.item())
        self._kmeans_init_prototypes_first_year(model, inputs)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        args.logger.info("[*] Year " + str(args.year) + " Training start")
        lowest_validation_loss = 1e7
        lowest_validation_loss_full = 1e7
        best_model_state = None
        best_model_loss_str = None
        counter = 0
        patience = int(
            getattr(
                args,
                "patience",
                getattr(args, "patience_later_years", getattr(args, "patience_first_year", 15)),
            )
        )

        total_time = 0.0
        use_time = []
        last_epoch = -1
        for epoch in range(args.epoch):
            last_epoch = epoch

            start_time = datetime.now()
            training_loss = self._train_one_epoch(model, train_loader, optimizer, lossfunc, epoch)
            epoch_time = (datetime.now() - start_time).total_seconds()
            total_time += epoch_time
            use_time.append(epoch_time)

            validation_loss = self._validate(model, val_loader, args.sub_adj, use_incremental_mapping=True)
            validation_loss_full = None
            if args.year > args.begin_year:
                validation_loss_full = self._validate(model, val_loader_full, args.adj, use_incremental_mapping=False)
                args.logger.info(
                    f"epoch:{epoch}, training loss:{training_loss:.4f} "
                    f"validation loss:{validation_loss:.4f} validation_full loss:{validation_loss_full:.4f}"
                )
            else:
                args.logger.info(
                    f"epoch:{epoch}, training loss:{training_loss:.4f} "
                    f"validation loss:{validation_loss:.4f}"
                )

            use_validation_full = args.year > args.begin_year
            monitor_loss = validation_loss_full if use_validation_full else validation_loss
            lowest_monitor_loss = lowest_validation_loss_full if use_validation_full else lowest_validation_loss

            validation_improved = monitor_loss <= lowest_monitor_loss
            if validation_loss <= lowest_validation_loss:
                lowest_validation_loss = round(validation_loss, 4)
            if validation_loss_full is not None and validation_loss_full <= lowest_validation_loss_full:
                lowest_validation_loss_full = round(validation_loss_full, 4)

            if validation_improved:
                counter = 0
            else:
                counter += 1

            if use_validation_full:
                if validation_improved:
                    best_model_state = {k: v.clone() for k, v in gnn_model.state_dict().items()}
                    best_model_loss_str = str(round(validation_loss_full, 4))
            elif validation_improved:
                best_model_state = {k: v.clone() for k, v in gnn_model.state_dict().items()}
                best_model_loss_str = str(round(validation_loss, 4))

            if counter > patience:
                break

        best_model_path = self._save_best_model(path, best_model_state, best_model_loss_str, lowest_validation_loss, gnn_model)
        best_model = build_model(args)
        best_model.load_state_dict(torch.load(best_model_path, map_location=args.device)["model_state_dict"])
        best_model = best_model.to(args.device)


        self.test_model(best_model, test_loader)

        epoch_num = last_epoch + 1
        average_epoch_time = sum(use_time) / len(use_time) if use_time else 0.0
        gpu_memory = self._get_gpu_memory_stats()
        self.result["time_info"][args.year] = {
            "total_time": total_time,
            "average_time": average_epoch_time,
            "epoch_num": epoch_num,
            "gpu_memory": gpu_memory,
        }
        selected_best_loss = float(best_model_loss_str) if best_model_loss_str is not None else lowest_validation_loss
        args.logger.info(
            "year\t{}\tgpu_memory\tallocated_mb\t{:.2f}\treserved_mb\t{:.2f}\tmax_allocated_mb\t{:.2f}\tmax_reserved_mb\t{:.2f}".format(
                args.year,
                gpu_memory["allocated_mb"],
                gpu_memory["reserved_mb"],
                gpu_memory["max_allocated_mb"],
                gpu_memory["max_reserved_mb"],
            )
        )
        args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))
        return TrainingSummary(best_model_path, selected_best_loss, total_time, average_epoch_time, epoch_num)

    def test_model(self, model, testset):
        args = self.args
        model.eval()
        pred_ = []
        truth_ = []
        loss = 0.0
        with torch.no_grad():
            cn = 0
            for data in testset:
                data = data.to(args.device, non_blocking=self.pin_memory)
                pred = model(data, args.adj)
                loss += func.mse_loss(data.y, pred, reduction="mean")
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred_.append(pred.cpu().data.numpy())
                truth_.append(data.y.cpu().data.numpy())
                cn += 1
            loss = loss / cn
            args.logger.info("[*] loss:{:.4f}".format(loss))
            pred_ = np.concatenate(pred_, 0)
            truth_ = np.concatenate(truth_, 0)
            self.metric(truth_, pred_)
            return loss

    def metric(self, ground_truth, prediction):
        args = self.args
        args.logger.info("[*] year {}, testing".format(args.year))
        for horizon in [3, 6, 12]:
            mae = masked_mae_np(ground_truth[:, :, :horizon], prediction[:, :, :horizon], 0)
            rmse = masked_mse_np(ground_truth[:, :, :horizon], prediction[:, :, :horizon], 0) ** 0.5
            mape = masked_mape_np(ground_truth[:, :, :horizon], prediction[:, :, :horizon], 0)
            args.logger.info(f"T:{horizon:<2d} MAE {mae:>7.4f} RMSE {rmse:>7.4f} MAPE {mape:>7.4f}")
            self.result[horizon]["mae"][args.year] = mae
            self.result[horizon]["mape"][args.year] = mape
            self.result[horizon]["rmse"][args.year] = rmse
        return mae

    def _build_loss(self):
        if self.args.loss == "mse":
            return func.mse_loss
        raise ValueError(f"Unsupported loss: {self.args.loss}")

    def _build_loaders(self, inputs):
        args = self.args
        if args.year > args.begin_year:
            train_loader = DataLoader(
                TrafficDataset(
                    "",
                    "",
                    x=inputs["train_x"][:, :, args.subgraph.numpy()],
                    y=inputs["train_y"][:, :, args.subgraph.numpy()],
                    edge_index="",
                    mode="subgraph",
                    node_ids=args.subgraph,
                ),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
            val_loader = DataLoader(
                TrafficDataset(
                    "",
                    "",
                    x=inputs["val_x"][:, :, args.subgraph.numpy()],
                    y=inputs["val_y"][:, :, args.subgraph.numpy()],
                    edge_index="",
                    mode="subgraph",
                    node_ids=args.subgraph,
                ),
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
            val_loader_full = DataLoader(
                TrafficDataset(inputs, "val"),
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
            self._build_subgraph_adj()
        else:
            train_loader = DataLoader(
                TrafficDataset(inputs, "train"),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
            val_loader = DataLoader(
                TrafficDataset(inputs, "val"),
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
            val_loader_full = None
            vars(args)["sub_adj"] = vars(args)["adj"]

        test_loader = DataLoader(
            TrafficDataset(inputs, "test"),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        return train_loader, val_loader, val_loader_full, test_loader

    def _build_subgraph_adj(self):
        args = self.args
        sub_nodes = args.subgraph.to(args.device)
        if hasattr(args, "adj_physical"):
            sub_adj = args.adj_physical.detach().to(args.device)[sub_nodes][:, sub_nodes]
        else:
            graph = nx.Graph()
            graph.add_nodes_from(range(args.subgraph.size(0)))
            graph.add_edges_from(args.subgraph_edge_index.numpy().T)
            adj = nx.to_numpy_array(graph)
            sub_adj = torch.from_numpy(adj).to(torch.float).to(args.device)


        sub_adj = torch.clamp(sub_adj, min=0.0)
        row_sum = sub_adj.sum(dim=1, keepdim=True)
        sub_adj = sub_adj / (row_sum + 1e-6)

        vars(args)["sub_adj"] = sub_adj.detach()

    def _build_model(self, inputs, lossfunc):
        args = self.args
        if args.year > args.begin_year:
            gnn_model, _ = self.load_best_model()
            model = gnn_model
        else:
            gnn_model = build_model(args).to(args.device)
            model = gnn_model

        return model, gnn_model

    def _train_one_epoch(self, model, train_loader, optimizer, lossfunc, epoch):
        args = self.args
        model.train()
        training_loss = 0.0
        cn = 0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(args.device, non_blocking=self.pin_memory)
            self._clear_prototype_cached_tensors(model)
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)
            pred, target = self._prepare_incremental_batch(pred, data)
            loss = lossfunc(target, pred, reduction="mean")
            proto_reg_loss = self._prototype_regularization_loss(model)
            if proto_reg_loss is not None:
                loss += proto_reg_loss
            else:
                self._clear_prototype_cached_tensors(model)

            training_loss += float(loss.detach())
            loss.backward()
            self._clear_prototype_cached_tensors(model)
            optimizer.step()
            cn += 1

        return training_loss / cn

    def _clear_prototype_cached_tensors(self, model):
        base_model = model
        if hasattr(base_model, "clear_cached_tensors"):
            base_model.clear_cached_tensors()
        if hasattr(base_model, "prototype_compactness_loss"):
            base_model.prototype_compactness_loss = None
        if hasattr(base_model, "prototype") and hasattr(base_model.prototype, "last_attn"):
            base_model.prototype.last_attn = None

    def _prototype_regularization_loss(self, model):
        args = self.args
        compactness_weight = float(getattr(args, "prototype_compactness_weight", 0.0))
        separation_weight = float(getattr(args, "prototype_separation_weight", 0.0))
        if compactness_weight <= 0.0 and separation_weight <= 0.0:
            return None

        base_model = model
        if not hasattr(base_model, "prototype"):
            return None

        proto = base_model.prototype
        losses = []

        if compactness_weight > 0.0:
            compactness_loss = getattr(base_model, "prototype_compactness_loss", None)
            if compactness_loss is not None:
                base_model.prototype_compactness_loss = None
                losses.append(compactness_weight * compactness_loss)


        if separation_weight > 0.0 and hasattr(proto, "_prototype_matrix"):
            proto_mat = proto._prototype_matrix()
            if proto_mat is not None and proto_mat.ndim == 2 and proto_mat.size(0) > 1:
                proto_norm = func.normalize(proto_mat, p=2, dim=-1)
                sim = torch.matmul(proto_norm, proto_norm.t())
                off_diag = sim - torch.diag_embed(torch.diagonal(sim))
                separation_loss = (off_diag ** 2).sum() / (proto_mat.size(0) * (proto_mat.size(0) - 1))
                losses.append(separation_weight * separation_loss)

        if not losses:
            return None
        return sum(losses)

    def _validate(self, model, val_loader, adj, use_incremental_mapping):
        args = self.args
        model.eval()
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(args.device, non_blocking=self.pin_memory)
                pred = model(data, adj)
                if use_incremental_mapping:
                    pred, target = self._prepare_incremental_batch(pred, data)
                else:
                    target = data.y
                loss = masked_mae_np(target.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        return float(validation_loss / cn)

    def _prepare_incremental_batch(self, pred, data):
        args = self.args
        if args.year > args.begin_year:
            pred, _ = to_dense_batch(pred, batch=data.batch)
            target, _ = to_dense_batch(data.y, batch=data.batch)
            pred = pred[:, args.mapping, :]
            target = target[:, args.mapping, :]
            return pred, target
        return pred, data.y

    def _save_best_model(self, path, best_model_state, best_model_loss_str, lowest_validation_loss, gnn_model):
        ct.mkdirs(path)
        if best_model_state is not None:
            best_model_path = osp.normpath(osp.join(path, best_model_loss_str + ".pkl"))
            torch.save({"model_state_dict": best_model_state}, best_model_path)
        else:
            best_model_path = osp.normpath(osp.join(path, str(lowest_validation_loss) + ".pkl"))
            torch.save({"model_state_dict": gnn_model.state_dict()}, best_model_path)
        return best_model_path

    def _kmeans_init_prototypes_first_year(self, model, inputs):
        args = self.args
        if not getattr(args, "prototype_first_year_kmeans", True):
            return
        if args.year != args.begin_year:
            return

        base_model = model
        if not (hasattr(base_model, "prototype") and hasattr(base_model, "get_preprototype_features")):
            return

        proto = base_model.prototype
        k = int(proto.num_prototypes)
        max_samples = int(getattr(args, "prototype_kmeans_max_samples", 50000))
        iters = int(getattr(args, "prototype_kmeans_iters", 20))
        seed = int(getattr(args, "prototype_kmeans_seed", 13))

        train_loader = DataLoader(
            TrafficDataset(inputs, "train"),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

        feats = []
        total = 0
        base_model.eval()
        with torch.no_grad():
            for data in train_loader:
                data = data.to(args.device, non_blocking=self.pin_memory)
                if hasattr(base_model, "get_prototype_kmeans_features"):
                    x_res = base_model.get_prototype_kmeans_features(data, args.adj)
                else:
                    x_res = base_model.get_preprototype_features(data, args.adj)
                if x_res is None or x_res.numel() == 0:
                    continue
                x_res = x_res.detach()
                if total + x_res.size(0) > max_samples:
                    keep = max_samples - total
                    if keep <= 0:
                        break
                    x_res = x_res[:keep]
                feats.append(x_res)
                total += x_res.size(0)
                if total >= max_samples:
                    break

        if not feats:
            if hasattr(args, "logger"):
                args.logger.warning("[!] Prototype kmeans init skipped: no features collected")
            return

        data = torch.cat(feats, dim=0)
        if data.size(0) < k:
            repeat = (k + data.size(0) - 1) // data.size(0)
            data = data.repeat(repeat, 1)

        g = torch.Generator(device=data.device)
        g.manual_seed(seed)
        init_idx = torch.randperm(data.size(0), generator=g, device=data.device)[:k]
        centers = data[init_idx].clone()

        for _ in range(iters):
            dist = torch.cdist(data, centers, p=2)
            labels = torch.argmin(dist, dim=1)
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(k, device=data.device, dtype=torch.long)
            new_centers.index_add_(0, labels, data)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
            nonzero = counts > 0
            if nonzero.any():
                new_centers[nonzero] = new_centers[nonzero] / counts[nonzero].unsqueeze(1).to(new_centers.dtype)
            if (~nonzero).any():
                ridx = torch.randint(0, data.size(0), ((~nonzero).sum().item(),), generator=g, device=data.device)
                new_centers[~nonzero] = data[ridx]
            centers = new_centers

        with torch.no_grad():
            proto.A.copy_(centers)
        if hasattr(args, "logger"):
            args.logger.info(f"[*] First-year prototype kmeans init done (space=query, samples={data.size(0)}, k={k}, iters={iters})")
