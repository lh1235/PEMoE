# PEMoE

Code for **PEMoE**, a continual traffic flow forecasting model for evolving road networks. PEMoE combines graph convolutions, bidirectional node–prototype interaction, history-conditioned mixture-of-experts routing, and representative node selection for incremental training.

### Requirements

* Python = 3.10
* PyTorch = 2.1.0
* PyTorch Geometric = 2.4.0

```bash
conda env create -f environment.yml
conda activate pemoe
```

The Conda environment uses CUDA 11.8. For a CPU environment, install the dependencies with `pip install -r requirements.txt` and run with `--gpuid -1`.

### Data

Datasets are not included. The PeMS03 data released with TrafficStream can be downloaded from [Google Drive](https://drive.google.com/file/d/1P5wowSaNSWBNCK3mQwESp-G2zsutXc5S/view). Prepare the corresponding evolving PeMS04 data separately. Organize the raw flow arrays and yearly graphs as follows:

```text
data/
├── PeMS03_Evolve/
│   ├── raw/          # 2011.npz, ..., 2017.npz
│   └── graph/        # 2011_adj.npz, ..., 2017_adj.npz
└── PeMS4_Evolve/
    ├── raw/          # 0.npz, ..., 6.npz
    └── graph/        # 0_adj.npz, ..., 6_adj.npz
```

Each `.npz` file must contain an array named `x`: raw traffic has shape `[time_steps, nodes]`, and adjacency has shape `[nodes, nodes]`. Node IDs must remain consistent across stages, with new nodes appended after existing nodes. For TrafficStream-style folders, use `finaldata/` as `raw/`; for PeMS04, use the corresponding `finaldata_removed/` split and matching graphs. Alternatively, edit the data paths in the configuration files. Relative paths are resolved from this repository's root.

Both configurations build caches in `processed/` on the first run (`data_process=1`). Set `data_process=0` to reuse them. Preprocessing uses the first 31 days of each stage with chronological 60%/20%/20% train/validation/test splits. As in the source implementation, input windows are z-scored independently within each split; targets remain in raw units.

### Usages

Run commands from the repository root.

* PEMoE on PeMS03-Evolve:

```bash
python main.py --conf conf/pems03.json --gpuid 0
```

* PEMoE on PeMS04-Evolve:

```bash
python main.py --conf conf/pems04.json --gpuid 0
```

Use `--gpuid -1` for CPU execution. The default configuration is `conf/pems03.json`.

Both settings use 12 input steps and 12 prediction steps, hidden width 128, 4 prototypes, and 4 experts with Top-2 routing weights. All four expert convolutions execute before their outputs are mixed. Training uses AdamW with learning rate 0.001, batch size 128, up to 200 epochs, patience 10, and seed 13.

The first stage trains on the full graph. Each later stage inherits the preceding best checkpoint, includes all new nodes, and fills a 20% seed budget with representative nodes. Mandatory new nodes may exceed that budget. Two-hop physical neighbors provide context, while forecast supervision applies to seed nodes. Full-graph validation MAE selects checkpoints; node history descriptors use training observations only.

Checkpoints, logs, node selections, and `metrics.json` are written under `outputs/<dataset>/<run>/`. Metrics report cumulative MAE, RMSE, and MAPE over the first 3, 6, and 12 prediction steps. Data and generated outputs are excluded by `.gitignore`.
