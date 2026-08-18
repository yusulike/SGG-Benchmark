#!/usr/bin/env python3
"""
Simple evaluation script for Scene Graph Generation with Hydra configs.

The script automatically loads the config from the checkpoint directory.

Usage:
    # Evaluate using checkpoint folder (auto-detects best/final model)
    python tools/relation_eval_hydra.py \
        --checkpoint-dir checkpoints/VG/triplet_attention_yolo12m \
        --task sgdet

    # Evaluate specific checkpoint file
    python tools/relation_eval_hydra.py \
        --checkpoint checkpoints/VG/transformer-yolo12m-v2/model_epoch_4.pth \
        --task sgcls

    # Evaluate on different split
    python tools/relation_eval_hydra.py \
        --checkpoint-dir checkpoints/VG/react \
        --task predcls \
        --test-split custom_dataset_val

    # Save predictions and visualizations
    python tools/relation_eval_hydra.py \
        --checkpoint-dir checkpoints/VG/model \
        --task sgdet \
        --save-predictions \
        --visualize
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sgg_benchmark.config.hydra_config import Config, load_config_from_file
from sgg_benchmark.data import make_data_loader, get_dataset_statistics
from sgg_benchmark.engine.inference import inference
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.comm import synchronize, get_rank
from sgg_benchmark.utils.logger import setup_logger
from sgg_benchmark.utils.miscellaneous import mkdir, set_seed


def find_config_file(checkpoint_dir):
    """Find the config file in a checkpoint directory (also searches one level deep)"""
    checkpoint_dir = Path(checkpoint_dir)

    # Direct match first, then one level deep (e.g. HF download layout: <run-dir>/yolo12m/config.yml)
    candidates = [
        checkpoint_dir / "config.yaml",
        checkpoint_dir / "config.yml",
        checkpoint_dir / "hydra_config.yaml",
        checkpoint_dir / ".hydra" / "config.yaml",
    ]
    candidates += sorted(checkpoint_dir.glob("*/config.yaml")) + sorted(checkpoint_dir.glob("*/config.yml"))

    for config_path in candidates:
        if config_path.exists():
            print(f"Found config at: {config_path}")
            return str(config_path)

    raise FileNotFoundError(
        f"Could not find config in {checkpoint_dir}. "
        f"Looked in: {[str(p) for p in candidates]}"
    )


def find_checkpoint_file(checkpoint_dir):
    """Find the best or final checkpoint in directory (also searches one level deep)"""
    checkpoint_dir = Path(checkpoint_dir)

    # Priority order: best > final > latest epoch
    names = ["model_best.pth", "model_final.pth", "best_model.pth"]
    candidates = [checkpoint_dir / n for n in names]
    candidates += sorted(p for n in names for p in checkpoint_dir.glob(f"*/{n}"))

    # Check for explicit checkpoint files
    for ckpt in candidates:
        if ckpt.exists():
            print(f"Found checkpoint: {ckpt}")
            return str(ckpt)

    # Look for epoch checkpoints and use the latest
    epoch_ckpts = sorted(checkpoint_dir.glob("model_epoch_*.pth"))
    epoch_ckpts += sorted(checkpoint_dir.glob("*/model_epoch_*.pth"))
    if epoch_ckpts:
        latest = epoch_ckpts[-1]
        print(f"Found latest epoch checkpoint: {latest}")
        return str(latest)

    raise FileNotFoundError(
        f"Could not find any checkpoint in {checkpoint_dir}. "
        f"Looked for: {', '.join(names)}, model_epoch_*.pth (direct or one level deep)"
    )


def load_hydra_config(config_file):
    """Load Hydra config and wrap with YACS-like interface"""
    # Use load_config_from_file to ensure defaults are applied first
    hydra_cfg = load_config_from_file(config_file)
    
    # Wrap Hydra config with Config class for YACS-like attribute access
    cfg = Config(hydra_cfg)
    
    return cfg, hydra_cfg


def assert_task_mode(cfg, task):
    """Set config flags based on task mode"""
    # Note: cfg is a Config wrapper, we need to access the underlying DictConfig
    cfg_dict = cfg.cfg if hasattr(cfg, 'cfg') else cfg._cfg
    
    # Set flags based on task
    if task == "sgdet":
        cfg_dict.model.roi_relation_head.use_gt_box = False
        cfg_dict.model.roi_relation_head.use_gt_object_label = False
        print(f"Task={task}: Full detection mode")
    elif task == "sgcls":
        cfg_dict.model.roi_relation_head.use_gt_box = True
        cfg_dict.model.roi_relation_head.use_gt_object_label = False
        print(f"Task={task}: Using GT boxes")
    elif task == "predcls":
        cfg_dict.model.roi_relation_head.use_gt_box = True
        cfg_dict.model.roi_relation_head.use_gt_object_label = True
        print(f"Task={task}: Using GT boxes and labels")


def enable_inplace_relu(model):
    """Enable in-place ReLU for memory efficiency"""
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, torch.nn.ReLU(inplace=True))
        else:
            enable_inplace_relu(module)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SGG Model")
    
    # Checkpoint specification (provide one of these)
    parser.add_argument("--run-dir", type=str,
                        help="Run directory containing checkpoint and config, "
                             "same as --checkpoint-dir (also searches one level deep, "
                             "e.g. after downloading weights from the HuggingFace hub)")
    parser.add_argument("--checkpoint-dir", type=str,
                        help="Directory containing checkpoint and config (auto-detects best/final)")
    parser.add_argument("--checkpoint", type=str,
                        help="Specific checkpoint file path (relative paths are also "
                             "tried against the run/checkpoint directory)")
    parser.add_argument("--config-file", type=str,
                        help="Config file (if not in checkpoint dir)")
    
    # Evaluation settings
    parser.add_argument("--task", type=str, default="sgdet",
                        choices=["predcls", "sgcls", "sgdet"],
                        help="Evaluation task mode")
    parser.add_argument("--test-split", type=str, default=None,
                        help="Dataset split to evaluate (default: use config)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: checkpoint_dir/inference)")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save prediction results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    
    # Hardware settings
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    
    args = parser.parse_args()
    
    # ==================== Setup ====================
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    distributed = num_gpus > 1
    
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    # ==================== Load Config and Checkpoint ====================
    
    # Determine checkpoint and config paths (--run-dir is an alias of --checkpoint-dir)
    run_dir = args.run_dir or args.checkpoint_dir
    if run_dir:
        checkpoint_dir = Path(run_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        if args.checkpoint:
            checkpoint_path = args.checkpoint
            if not Path(checkpoint_path).exists():
                # relative --checkpoint refers to the run directory (as documented in the README)
                candidate = checkpoint_dir / args.checkpoint
                if candidate.exists():
                    checkpoint_path = str(candidate)
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            checkpoint_path = find_checkpoint_file(checkpoint_dir)
        config_file = args.config_file or find_config_file(checkpoint_dir)
    elif args.checkpoint:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Try to find config in parent directory
        checkpoint_dir = Path(checkpoint_path).parent
        config_file = args.config_file or find_config_file(checkpoint_dir)
    else:
        raise ValueError("Must provide one of --run-dir, --checkpoint-dir or --checkpoint")
    
    print("=" * 80)
    print("EVALUATION SETUP")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config:     {config_file}")
    print(f"Task:       {args.task}")
    print("=" * 80)
    
    # Load config
    cfg, hydra_cfg = load_hydra_config(config_file)
    
    # Override task mode
    assert_task_mode(cfg, args.task)
    
    # Override test split if specified
    if args.test_split:
        # Access the underlying DictConfig for modification
        cfg_dict = cfg.cfg if hasattr(cfg, 'cfg') else cfg._cfg
        cfg_dict.datasets.test = [args.test_split]
        print(f"Overriding test split: {args.test_split}")
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(checkpoint_dir / f"inference_{args.task}")
    mkdir(output_dir)
    
    # Set output dir in config
    cfg_dict = cfg.cfg if hasattr(cfg, 'cfg') else cfg._cfg
    cfg_dict.output_dir = output_dir
    
    # Set seed
    seed = cfg.SEED if hasattr(cfg, 'SEED') else (cfg.seed if hasattr(cfg, 'seed') else 42)
    set_seed(seed)
    
    # Setup logger
    verbose = cfg.VERBOSE if hasattr(cfg, 'VERBOSE') else (cfg.verbose if hasattr(cfg, 'verbose') else "INFO")
    logger = setup_logger("sgg_benchmark", output_dir, get_rank(), 
                         filename=f"eval_{args.task}.log", verbose=verbose)
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(f"Evaluating: {checkpoint_path}")
    logger.info(f"Task mode: {args.task}")
    
    # ==================== Build Model ====================
    logger.info("Building model...")
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    enable_inplace_relu(model)
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(checkpoint_path)
    
    # Load text embeddings if using YOLO World
    if "world" in cfg.MODEL.BACKBONE.TYPE:
        logger.info("Loading text embeddings for YOLO World...")
        stats = get_dataset_statistics(cfg)
        obj_classes = stats['obj_classes'][1:]
        model.backbone.load_txt_feats(obj_classes)
    
    # Set to eval mode
    model.backbone.eval()
    model.roi_heads.eval()
    
    # ==================== Setup Evaluation ====================
    iou_types = ("bbox",)
    
    # Check if RELATION_ON (handles both uppercase YACS and lowercase Hydra)
    relation_on = getattr(cfg.MODEL, 'RELATION_ON', None) or getattr(cfg, 'relation_on', False) or \
                  (hasattr(cfg, 'model') and getattr(cfg.model, 'relation_on', False))
    attribute_on = getattr(cfg.MODEL, 'ATTRIBUTE_ON', None) or getattr(cfg, 'attribute_on', False) or \
                   (hasattr(cfg, 'model') and getattr(cfg.model, 'attribute_on', False))
    
    if relation_on:
        logger.info("Evaluating relations")
        iou_types = iou_types + ("relations",)
    if attribute_on:
        logger.info("Evaluating attributes")
        iou_types = iou_types + ("attributes",)
    
    # Get dataset names — always read from the raw DictConfig so we avoid
    # the Config wrapper's case-insensitive magic returning default empty tuples.
    raw_cfg = cfg._cfg
    datasets_cfg = raw_cfg.get('datasets', {})
    dataset_names = list(datasets_cfg.get('test', []) or [])
    # Fallback: derive from datasets.name (e.g. "PSG" → "PSG_test")
    if not dataset_names:
        ds_name = datasets_cfg.get('name', '')
        if ds_name:
            dataset_names = [f"{ds_name}_test"]
    logger.info(f"Evaluating on dataset(s): {dataset_names}")
    
    # Create data loaders
    # dataset_to_test should be 'train', 'val', or 'test', not the actual dataset name
    # For evaluation, we typically use 'test'
    data_loaders_val = make_data_loader(
        cfg=cfg, 
        mode="test", 
        is_distributed=distributed,
        dataset_to_test='test'  # Use 'test' mode, not the dataset name
    )
    
    # ==================== Run Evaluation ====================
    # Check DTYPE (handles both formats)
    dtype = getattr(cfg, 'DTYPE', None) or getattr(cfg, 'dtype', 'float32')
    use_amp = dtype == "float16" or args.amp
    
    # Get detections_per_img
    detections_per_img = 100  # default
    if hasattr(cfg, 'MODEL') and hasattr(cfg.MODEL, 'ROI_HEADS'):
        detections_per_img = getattr(cfg.MODEL.ROI_HEADS, 'DETECTIONS_PER_IMG', 100)
    elif hasattr(cfg, 'model') and hasattr(cfg.model, 'roi_heads'):
        detections_per_img = getattr(cfg.model.roi_heads, 'detections_per_img', 100)
    
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating on: {dataset_name}")
        logger.info(f"{'='*80}\n")
        
        output_folder = os.path.join(
            output_dir, 
            f"{dataset_name}_{args.task}_det{detections_per_img}"
        )
        mkdir(output_folder)
        
        # Get config values for inference (handles both formats)
        rpn_only = getattr(cfg.MODEL, 'RPN_ONLY', False) if hasattr(cfg, 'MODEL') else False
        device = getattr(cfg.MODEL, 'DEVICE', 'cuda') if hasattr(cfg, 'MODEL') else \
                 (getattr(cfg.model, 'device', 'cuda') if hasattr(cfg, 'model') else 'cuda')
        
        # Get test expectations
        expected_results = []
        expected_sigma = 4
        if hasattr(cfg, 'TEST'):
            expected_results = getattr(cfg.TEST, 'EXPECTED_RESULTS', [])
            expected_sigma = getattr(cfg.TEST, 'EXPECTED_RESULTS_SIGMA_TOL', 4)
        elif hasattr(cfg, 'test'):
            expected_results = getattr(cfg.test, 'expected_results', [])
            expected_sigma = getattr(cfg.test, 'expected_results_sigma_tol', 4)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            results = inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=rpn_only,
                device=device,
                expected_results=expected_results,
                expected_results_sigma_tol=expected_sigma,
                output_folder=output_folder if args.save_predictions else None,
                logger=logger,
                informative=True,
            )
        
        synchronize()
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION COMPLETE: {dataset_name}")
        logger.info(f"Results saved to: {output_folder}")
        logger.info(f"{'='*80}\n")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    logger.info("Evaluation finished!")


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
