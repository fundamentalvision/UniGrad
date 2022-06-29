import argparse
from pathlib import Path 

import yaml
from easydict import EasyDict as edict

from project.utils import init_environ
from project.pretrainer import Pretrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='path to config')
    
    args = parser.parse_args()
    return args


def main():
    # parse config
    args = parse_args()
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = edict(cfg)
    cfg.config_path = args.config_path

    # init environment
    logger = init_environ(cfg)

    # build pretrainer
    pretrainer = Pretrainer(cfg, logger)
    pretrainer.pretrain()


if __name__ == '__main__':
    main()