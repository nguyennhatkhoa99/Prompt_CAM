#!/usr/bin/env python3
"""
Training script for Reidac9 dataset using Prompt_CAM.

Usage:
    # Train with default config (DINO backbone)
    python train_reidac9.py

    # Train with custom learning rate and epochs
    python train_reidac9.py --lr 0.01 --epoch 200

    # Train with DINOv2 backbone
    python train_reidac9.py --model dinov2 --pretrained_weights vit_base_patch14_dinov2

    # Resume or use custom config
    python train_reidac9.py --config my_custom_config.yaml
"""

import argparse
import os
import sys
import time

from experiment.run import basic_run
from utils.setup_logging import get_logger
from utils.misc import set_seed, load_yaml, override_args_with_yaml

logger = get_logger('Prompt_CAM')

# Reidac9 class names for reference
CLASS_NAMES = {
    0: 'Exterior', 1: 'Interior', 2: 'another', 3: 'bathroom',
    4: 'bedroom', 5: 'dining_room', 6: 'document', 7: 'kitchen',
    8: 'living_room'
}


def setup_parser():
    parser = argparse.ArgumentParser(description='Train Prompt_CAM on Reidac9 Dataset')

    # YAML Config
    parser.add_argument('--config', type=str, default='train_config_reidac9.yaml',
                        help='Path to YAML config file (default: train_config_reidac9.yaml)')

    # Model
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        choices=['vit_base_patch16_224_in21k', 'vit_base_mae',
                                 'vit_base_patch14_dinov2', 'vit_base_patch16_dino',
                                 'vit_base_patch16_clip_224'],
                        help='Pretrained weights (overrides config)')
    parser.add_argument('--model', type=str, default=None, choices=['vit', 'dino', 'dinov2'],
                        help='Model backbone (overrides config)')
    parser.add_argument('--train_type', type=str, default=None,
                        choices=['vpt', 'prompt_cam', 'linear'],
                        help='Training type (overrides config)')
    parser.add_argument('--drop_path_rate', default=None, type=float)

    # Optimizer / Scheduler
    parser.add_argument('--optimizer', default=None, choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', default=None, type=float, help='Learning rate')
    parser.add_argument('--epoch', default=None, type=int, help='Number of epochs')
    parser.add_argument('--warmup_epoch', default=None, type=int)
    parser.add_argument('--lr_min', type=float, default=None)
    parser.add_argument('--warmup_lr_init', type=float, default=None)
    parser.add_argument('--wd', type=float, default=None, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--early_patience', type=int, default=None)

    # Data
    parser.add_argument('--data', default=None, help='Dataset name')
    parser.add_argument('--data_path', default=None, help='Path to dataset root')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--test_batch_size', default=None, type=int)
    parser.add_argument('--crop_size', default=None, type=int)

    # VPT
    parser.add_argument('--vpt_num', default=None, type=int, help='Number of visual prompts')
    parser.add_argument('--vpt_mode', type=str, default=None, choices=['deep', 'shallow'])
    parser.add_argument('--vpt_layer', default=None, type=int)
    parser.add_argument('--vpt_dropout', default=None, type=float)

    # Run settings
    parser.add_argument('--final_run', action='store_true', default=None)
    parser.add_argument('--full', action='store_true', default=None)
    parser.add_argument('--normalized', action='store_true', default=None)
    parser.add_argument('--store_ckp', action='store_true', default=None)
    parser.add_argument('--final_acc_hp', action='store_true', default=None)
    parser.add_argument('--debug', action='store_true', default=None)
    parser.add_argument('--gpu_num', default=None, type=int)
    parser.add_argument('--random_seed', default=None, type=int)
    parser.add_argument('--eval_freq', default=None, type=int)

    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    # Load YAML config first
    if args.config and os.path.exists(args.config):
        yaml_config = load_yaml(args.config)
        if yaml_config:
            args = override_args_with_yaml(args, yaml_config)
    else:
        logger.warning(f"Config file '{args.config}' not found, using defaults")

    # Override with any CLI arguments that were explicitly set
    cli_args = parser.parse_args()
    for key, value in vars(cli_args).items():
        if value is not None and key != 'config':
            setattr(args, key, value)

    set_seed(args.random_seed)
    args.vis_attn = False

    # Print training info
    logger.info("=" * 60)
    logger.info("Training Prompt_CAM on Reidac9 Dataset")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Pretrained weights: {args.pretrained_weights}")
    logger.info(f"Train type: {args.train_type}")
    logger.info(f"VPT num prompts: {args.vpt_num}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epoch}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Classes: {args.class_num} ({', '.join(CLASS_NAMES.values())})")
    logger.info("=" * 60)

    start = time.time()
    basic_run(args)
    end = time.time()

    logger.info(f'----------- Total training time: {(end - start) / 60:.2f} mins -----------')
    logger.info(f'Checkpoint saved in: ./output/')


if __name__ == '__main__':
    main()
