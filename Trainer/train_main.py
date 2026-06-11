"""
train_main.py
=============
Single entry point for the Workflow Orchestrator training module.
Supports YOLO and HuggingFace training backends via factory patterns

Usage:
    # YOLO detection
    python train_main.py --framework yolo --model yolov8n.pt --task detect --data coco128.yaml --output_path /path/to/outputs

    # YOLO segmentation
    python train_main.py --framework yolo --model yolov8n-seg.pt --task segment --data coco128-seg.yaml --output_path /path/to/outputs

    # YOLO classification
    python train_main.py --framework yolo --model yolov8n-cls.pt --task classify --data imagenet100 --output_path /path/to/outputs

    # HuggingFace classification
    python train_main.py --framework huggingface --model google/vit-base-patch16-224 --task classify --data /path/to/dataset --output_path /path/to/outputs

    # HuggingFace detection
    python train_main.py --framework huggingface --model facebook/detr-resnet-50 --task detect --data detection-datasets/coco --output_path /path/to/outputs
"""

import argparse
import sys
import torch

from factory import TrainerFactory


# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────

DEFAULT_OUTPUT          = "outputs"
DEFAULT_EPOCHS          = 10
DEFAULT_BATCH           = 16
DEFAULT_DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED            = 42
DEFAULT_WORKERS         = 4
DEFAULT_PATIENCE        = 3

# YOLO specific
DEFAULT_IMGSZ           = 640
DEFAULT_LR0             = 0.01
DEFAULT_LRF             = 0.01
DEFAULT_MOMENTUM        = 0.937
DEFAULT_WEIGHT_DECAY    = 0.0005
DEFAULT_WARMUP_EPOCHS   = 3.0
DEFAULT_WARMUP_MOMENTUM = 0.8
DEFAULT_WARMUP_BIAS_LR  = 0.1
DEFAULT_OPTIMIZER       = "auto"
DEFAULT_DROPOUT         = 0.0
DEFAULT_FRACTION        = 1.0
DEFAULT_CLOSE_MOSAIC    = 10
DEFAULT_SAVE_PERIOD     = -1

# HuggingFace specific
DEFAULT_LR              = 2e-5
DEFAULT_HF_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS    = 500
DEFAULT_LR_SCHEDULER    = "linear"
DEFAULT_HF_OPTIMIZER    = "adamw_torch"
DEFAULT_EVAL_STRATEGY   = "epoch"
DEFAULT_SAVE_STRATEGY   = "epoch"
DEFAULT_SAVE_TOTAL      = 2
DEFAULT_LOGGING_STEPS   = 50
DEFAULT_GRADIENT_ACCUM  = 1
DEFAULT_NPROC_PER_NODE  = 1
DEFAULT_NUM_NODES       = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflow Orchestrator — Training Module",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Required ──────────────────────────────
    parser.add_argument("--framework",   type=str, required=True,
                        choices=TrainerFactory.list_frameworks(),
                        help="Training framework to use")
    parser.add_argument("--model",       type=str, required=True,
                        help="Model name or path. "
                             "YOLO: yolov8n.pt, yolov9c.pt etc. "
                             "HuggingFace: google/vit-base-patch16-224 etc.")
    parser.add_argument("--task",        type=str, required=True,
                        choices=["detect", "segment", "classify"],
                        help="Task type")
    parser.add_argument("--data",        type=str, required=True,
                        help="Dataset path or name. "
                             "YOLO: path to .yaml or folder. "
                             "HuggingFace: local folder or HuggingFace Hub ID.")

    # ── Output ────────────────────────────────
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT,
                        help="Root directory for all outputs")
    parser.add_argument("--name",        type=str, default=None,
                        help="Experiment name (auto-generated if not set)")

    # ── Common Training ───────────────────────
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch",       type=int,   default=DEFAULT_BATCH,
                        help="Batch size per device")
    parser.add_argument("--device",      type=str,   default=DEFAULT_DEVICE,
                        help="Device: cuda or cpu")
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED,
                        help="Random seed")
    parser.add_argument("--workers",     type=int,   default=DEFAULT_WORKERS,
                        help="Dataloader workers")
    parser.add_argument("--patience",    type=int,   default=DEFAULT_PATIENCE,
                        help="Early stopping patience")

    # ── Distributed Training ──────────────────
    parser.add_argument("--nproc_per_node", type=int, default=DEFAULT_NPROC_PER_NODE,
                        help="Number of GPUs per node")
    parser.add_argument("--num_nodes",      type=int, default=DEFAULT_NUM_NODES,
                        help="Number of nodes")
    parser.add_argument("--master_addr",    type=str, default="localhost",
                        help="Master node address")
    parser.add_argument("--master_port",    type=str, default="29500",
                        help="Master node port")
    parser.add_argument("--backend",        type=str, default="nccl",
                        choices=["nccl", "gloo", "mpi"],
                        help="Distributed backend")

    # ── YOLO Hyperparameters ──────────────────
    yolo_group = parser.add_argument_group("YOLO hyperparameters")
    yolo_group.add_argument("--imgsz",           type=int,   default=DEFAULT_IMGSZ)
    yolo_group.add_argument("--lr0",             type=float, default=DEFAULT_LR0)
    yolo_group.add_argument("--lrf",             type=float, default=DEFAULT_LRF)
    yolo_group.add_argument("--momentum",        type=float, default=DEFAULT_MOMENTUM)
    yolo_group.add_argument("--weight_decay",    type=float, default=DEFAULT_WEIGHT_DECAY)
    yolo_group.add_argument("--warmup_epochs",   type=float, default=DEFAULT_WARMUP_EPOCHS)
    yolo_group.add_argument("--warmup_momentum", type=float, default=DEFAULT_WARMUP_MOMENTUM)
    yolo_group.add_argument("--warmup_bias_lr",  type=float, default=DEFAULT_WARMUP_BIAS_LR)
    yolo_group.add_argument("--optimizer",       type=str,   default=DEFAULT_OPTIMIZER)
    yolo_group.add_argument("--cos_lr",          action="store_true", default=False)
    yolo_group.add_argument("--dropout",         type=float, default=DEFAULT_DROPOUT)
    yolo_group.add_argument("--fraction",        type=float, default=DEFAULT_FRACTION)
    yolo_group.add_argument("--freeze",          type=int,   default=None)
    yolo_group.add_argument("--close_mosaic",    type=int,   default=DEFAULT_CLOSE_MOSAIC)
    yolo_group.add_argument("--save_period",     type=int,   default=DEFAULT_SAVE_PERIOD)
    yolo_group.add_argument("--amp",             action="store_true", default=True)
    yolo_group.add_argument("--resume",          action="store_true", default=False)
    yolo_group.add_argument("--profile",         action="store_true", default=False)

    # ── HuggingFace Hyperparameters ───────────
    hf_group = parser.add_argument_group("HuggingFace hyperparameters")
    hf_group.add_argument("--lr",                          type=float, default=DEFAULT_LR)
    hf_group.add_argument("--hf_weight_decay",             type=float, default=DEFAULT_HF_WEIGHT_DECAY)
    hf_group.add_argument("--warmup_steps",                type=int,   default=DEFAULT_WARMUP_STEPS)
    hf_group.add_argument("--lr_scheduler",                type=str,   default=DEFAULT_LR_SCHEDULER,
                          choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"])
    hf_group.add_argument("--hf_optimizer",                type=str,   default=DEFAULT_HF_OPTIMIZER)
    hf_group.add_argument("--eval_strategy",               type=str,   default=DEFAULT_EVAL_STRATEGY,
                          choices=["no", "steps", "epoch"])
    hf_group.add_argument("--save_strategy",               type=str,   default=DEFAULT_SAVE_STRATEGY,
                          choices=["no", "steps", "epoch"])
    hf_group.add_argument("--save_total_limit",            type=int,   default=DEFAULT_SAVE_TOTAL)
    hf_group.add_argument("--logging_steps",               type=int,   default=DEFAULT_LOGGING_STEPS)
    hf_group.add_argument("--gradient_accumulation_steps", type=int,   default=DEFAULT_GRADIENT_ACCUM)
    hf_group.add_argument("--fp16",                        action="store_true", default=False)
    hf_group.add_argument("--train_samples",               type=int,   default=100)
    hf_group.add_argument("--val_samples",                 type=int,   default=20)

    return parser.parse_args()


def print_config(args):
    """Print training configuration."""
    print("\n" + "=" * 50)
    print("WORKFLOW ORCHESTRATOR — TRAINING MODULE")
    print("=" * 50)
    print(f"\n  Framework   : {args.framework}")
    print(f"  Model       : {args.model}")
    print(f"  Task        : {args.task}")
    print(f"  Dataset     : {args.data}")
    print(f"  Output Path : {args.output_path}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch}")
    print(f"  Device      : {args.device}")
    total_gpus = args.nproc_per_node * args.num_nodes
    if total_gpus > 1:
        print(f"  Total GPUs  : {total_gpus} ({args.num_nodes} nodes x {args.nproc_per_node} GPUs)")
    print()


if __name__ == "__main__":
    args = parse_args()

    # Fix weight_decay naming for HuggingFace
    # HuggingFace trainer uses args.weight_decay internally
    if args.framework == "huggingface":
        args.weight_decay = args.hf_weight_decay
        args.optimizer    = args.hf_optimizer

    print_config(args)

    # Create the right trainer via factory
    trainer = TrainerFactory.create(args.framework, args)

    # Run full training pipeline
    trainer.run()
