# -*- coding: utf-8 -*-
import argparse
import logging
import os.path as osp
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from utils import common_tools as ct


class ExperimentConfig:
    """Load CLI + JSON config and attach runtime-only fields."""

    @staticmethod
    def build_parser():
        parser = argparse.ArgumentParser(description="Train PEMoE on an evolving traffic network.")
        parser.add_argument("--conf", type=str, default="conf/pems03.json")
        parser.add_argument("--gpuid", type=int, default=0, help="CUDA device index; -1 selects CPU")
        parser.add_argument("--logname", type=str, default=None)
        parser.add_argument(
            "--load_first_year",
            type=int,
            choices=(0, 1),
            default=None,
            help="0: training first year, 1: load from model path of first year",
        )
        parser.add_argument(
            "--first_year_model_path",
            type=str,
            default=None,
            help="path to a compatible pretrained .pkl checkpoint",
        )
        return parser

    @classmethod
    def from_cli(cls):
        args = cls.build_parser().parse_args()
        cls.load_json(args)
        cls.seed_everything(int(getattr(args, "seed", 13)))
        cls.attach_device(args)
        return args

    @staticmethod
    def load_json(args):
        root_dir = Path(__file__).resolve().parents[1]
        conf_path = root_dir / args.conf
        info = ct.load_json_file(str(conf_path))
        if 'sweep' in info:
            raise ValueError('Use a single baseline configuration without a sweep')
        overrides = {key: getattr(args, key) for key in ('logname', 'load_first_year', 'first_year_model_path')
                     if getattr(args, key, None) is not None}
        info["time"] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")

        for key, value in info.items():
            if key != "gpuid":
                vars(args)[key] = value
        vars(args).update(overrides)
        args.load_first_year = int(args.load_first_year or 0)

        if args.first_year_model_path:
            args.first_year_model_path = str((root_dir / args.first_year_model_path).resolve())
        if args.load_first_year or not args.train:
            if not args.first_year_model_path or not Path(args.first_year_model_path).is_file():
                raise ValueError('Provide --first_year_model_path pointing to a compatible checkpoint')

        for key in ('raw_data_path', 'save_data_path', 'graph_path', 'model_path'):
            setattr(args, key, str((root_dir / getattr(args, key)).resolve()))
        vars(args)["path"] = osp.join(args.model_path, args.logname + args.time)
        ct.mkdirs(args.path)
        import json
        with open(osp.join(args.path, 'config.json'), 'w', encoding='utf-8') as stream:
            recorded = dict(info)
            for key in ('gpuid', 'logname', 'load_first_year', 'first_year_model_path',
                        'raw_data_path', 'save_data_path', 'graph_path', 'model_path'):
                recorded[key] = getattr(args, key)
            json.dump(recorded, stream, indent=2)
        return args

    @staticmethod
    def attach_device(args):
        device = torch.device(f"cuda:{args.gpuid}") if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
        vars(args)["device"] = device
        return device

    @staticmethod
    def seed_everything(seed=0):
        max_seed = (1 << 32) - 1
        random.seed(seed)
        np.random.seed(random.randint(0, max_seed))
        torch.manual_seed(random.randint(0, max_seed))
        torch.cuda.manual_seed(random.randint(0, max_seed))
        torch.cuda.manual_seed_all(random.randint(0, max_seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def init_logger(args):
        log_dir, log_filename = args.path, args.logname
        ct.mkdirs(log_dir)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(message)s")

        fh = logging.FileHandler(osp.join(log_dir, log_filename + ".log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info("logger name:%s", osp.join(log_dir, log_filename + ".log"))
        vars(args)["logger"] = logger
        return logger
