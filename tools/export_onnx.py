#!/usr/bin/env python3
"""Export a trained SGG model to ONNX optimized for real-time inference.

Usage:
    python tools/export_onnx.py --run-dir checkpoints/IndoorVG/react_stable_v1 --onnx-path model.onnx --device cuda
"""
import os
import argparse
import torch
from omegaconf import OmegaConf, open_dict
from pathlib import Path
import subprocess
import sys

try:
    import onnx
    from onnx import helper, TensorProto
except Exception:
    onnx = None

from sgg_benchmark.utils.env import setup_environment
from sgg_benchmark.config import get_cfg
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer

setup_environment()


def export(model, sample_input, onnx_path, is_yolo=False, opset=17):
    # The webcam demo always letterboxes to the model's fixed input size (e.g. 640×640),
    # so spatial dimensions can be static.  Only batch size and output counts are dynamic.
    dynamic_axes = {'input': {0: 'batch_size'}}
    output_names = ['output']
    if is_yolo:
        output_names = ['boxes', 'rels']
        dynamic_axes['boxes'] = {0: 'num_objects'}
        dynamic_axes['rels'] = {0: 'num_rels'}

    # PyTorch ≥2.6: torch.onnx.export defaults to dynamo=True (torch.export symbolic
    # tracing), which fails on data-dependent shapes in ultralytics NMS.
    # Using fallback=True makes the new exporter run first, contaminating YOLO head's
    # self.shape with SymInt — then the legacy jit.trace fails too.
    # Fix: call torch.onnx.utils.export directly (the pre-2.6 legacy TorchScript-based
    # ONNX exporter).  It uses torch.jit.trace with concrete values and never runs
    # torch.export, so no SymInt contamination occurs.
    # Note: dynamic_axes must NOT include height/width for the input — declaring those
    # dimensions as dynamic causes adaptive_avg_pool2d to fail ("input size not accessible")
    # in the legacy exporter because it needs concrete spatial dims to compute kernel size.
    torch.onnx.utils.export(
        model,
        (sample_input,),
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, help='Directory containing config.yml and checkpoint files')
    parser.add_argument('--onnx-path', default=None, help='Output ONNX path (default: <run-dir>/model.onnx)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--image', default=None, help='Optional: path to a single image to use for export (overrides dataset sample)')
    parser.add_argument('--glove-dir', default=None, help='Override the glove_dir in the config (useful when training was done on a remote cluster)')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if args.onnx_path is None:
        args.onnx_path = str(run_dir / 'model.onnx')

    config_file = run_dir / 'config.yml'
    if not config_file.exists():
        # also accept hydra yaml names
        alt = run_dir / 'hydra_config.yaml'
        if alt.exists():
            config_file = alt
        else:
            raise FileNotFoundError(f"Config file not found in run dir: expected config.yml or hydra_config.yaml in {run_dir}")

    yaml_cfg = OmegaConf.load(str(config_file))
    cfg = get_cfg(yaml_cfg)

    # ── Patch inaccessible paths (typical when exporting on a different machine) ──
    # glove_dir: prefer --glove-dir arg, then auto-detect from known locations
    def _resolve_glove_dir(cfg, override=None):
        candidates = []
        if override:
            candidates.append(override)
        # project root (the GloVe .txt / .pt files often live there)
        candidates.append(str(Path(__file__).parent.parent.resolve()))
        candidates.append(os.getcwd())

        glove_types = ['glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']
        for cand in candidates:
            for gt in glove_types:
                if os.path.isfile(os.path.join(cand, gt + '.txt')) or \
                   os.path.isfile(os.path.join(cand, gt + '.pt')):
                    return cand
        return None

    current_glove_dir = getattr(cfg, 'glove_dir', None)
    if args.glove_dir or (current_glove_dir and not os.path.isdir(current_glove_dir)):
        resolved = _resolve_glove_dir(cfg, args.glove_dir)
        if resolved:
            print(f"Patching glove_dir: {current_glove_dir!r} → {resolved!r}")
            with open_dict(cfg):
                cfg.glove_dir = resolved
        else:
            print(f"Warning: could not find a local GloVe directory; "
                  f"model building may fail. Pass --glove-dir to specify one.")

    # Build model
    model = build_detection_model(cfg)
    device = torch.device(args.device)
    model.to(device)

    # ── Find & load checkpoint ────────────────────────────────────────────────
    checkpointer = DetectronCheckpointer(cfg, model)
    ckpt_candidates = []
    for fn in os.listdir(run_dir):
        if fn.endswith('.pth') and (fn.startswith('best_model_epoch') or fn.startswith('model_epoch') or fn.startswith('best_model')):
            ckpt_candidates.append(run_dir / fn)

    if not ckpt_candidates:
        for fn in os.listdir(run_dir):
            if fn.endswith('.pth'):
                ckpt_candidates.append(run_dir / fn)

    if not ckpt_candidates:
        last_file = run_dir / 'last_checkpoint'
        if last_file.exists():
            with open(last_file, 'r') as f:
                candidate = f.read().strip()
                if os.path.exists(candidate):
                    ckpt_candidates.append(Path(candidate))

    if not ckpt_candidates:
        raise FileNotFoundError(f"No checkpoint (.pth) found in {run_dir}")

    ckpt_candidates = sorted(ckpt_candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint_path = str(ckpt_candidates[0])
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpointer.load(checkpoint_path)
    model.eval()

    # YOLO-World txt_feats (CLIP text embeddings of the class prompts) is a plain
    # attribute, NOT part of the state dict, so loading the full SGG checkpoint
    # leaves it at its random init (norm ~38 instead of ~1.7) and all class
    # scores collapse to ~0.  Restore it from the backbone checkpoint the run
    # was trained from (backbone weights there are identical).
    bb_ckpt = getattr(cfg.model, 'pretrained_detector_ckpt', None)
    if bb_ckpt and os.path.isfile(str(bb_ckpt)):
        model.backbone.load(str(bb_ckpt))
        print(f"Restored YOLO-World txt_feats from: {bb_ckpt}")
    else:
        print(f"Warning: pretrained_detector_ckpt {bb_ckpt!r} not found — "
              f"YOLO-World class scores may be invalid for world backbones.")

    # ── Build sample input ────────────────────────────────────────────────────
    def _load_image_as_tensor(img_path: Path, device, target_size: int = 640):
        """Matches sgg_benchmark.data.transforms.transforms.LetterBox + ToTensorYOLO"""
        try:
            import cv2
        except ImportError:
            raise RuntimeError('opencv-python is required')
            
        img = cv2.imread(str(img_path)) # BGR
        h, w = img.shape[:2]
        r = min(target_size / h, target_size / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        
        # Resize
        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # Padding (Center)
        dw, dh = target_size - new_unpad[0], target_size - new_unpad[1]
        top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
        left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
        
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # ToTensorYOLO: BGR to RGB, HWC to CHW, / 255.0
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, HWC to CHW
        import numpy as np
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).float() / 255.0
        return tensor.unsqueeze(0).to(device)

    sample_img = None
    if args.image:
        # Fast path: use provided image directly — no dataset needed
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f'Provided image not found: {img_path}')
        img_size = int(getattr(getattr(cfg, 'input', None) or {}, 'min_size_test', None) or 640)
        sample_img = _load_image_as_tensor(img_path, device, target_size=img_size)
        print(f"Using provided image for export: {img_path}  shape={list(sample_img.shape)}")
    else:
        # Try to load from the dataset (requires dataset paths to be accessible)
        try:
            data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=False)
            if not data_loaders_val or len(data_loaders_val) == 0:
                raise RuntimeError('Empty data loader')
            dataset = data_loaders_val[0].dataset
            for i in range(min(10, len(dataset))):
                item = dataset[i]
                if isinstance(item, tuple) and isinstance(item[0], torch.Tensor):
                    sample_img = item[0].unsqueeze(0).to(device)
                    break
        except Exception as e:
            print(f'Could not load dataset ({e}). Re-run with --image <path> to provide a sample image.')
            raise

        if sample_img is None:
            raise RuntimeError('Could not retrieve a tensor sample from the dataset')

    # For GeneralizedYOLO the model often expects an ImageList wrapper; attempt to call directly
    is_yolo = hasattr(model, 'export')
    # If YOLO-style model supports export=True, enable it temporarily
    if is_yolo:
        try:
            model.export = True
            model.export_obj_thres = 0.05
        except Exception:
            pass

    # Warmup forward pass: materialises any LazyLinear modules (e.g. P5SceneContextExtractor)
    # and caches internal state (e.g. YOLO head shape) before ONNX tracing.
    print("Running warmup forward pass to materialise lazy modules ...")
    with torch.no_grad():
        _ = model(sample_img)

    print(f"Exporting ONNX to {args.onnx_path} (opset=17), is_yolo={is_yolo}")
    export(model, sample_img, args.onnx_path, is_yolo=is_yolo, opset=17)

    # Reset export flag if changed
    if is_yolo:
        try:
            model.export = False
        except Exception:
            pass

    # Post-process: make Split size constants dynamic (derive from runtime tensor shape)
    def make_split_dynamic(onnx_path):
        global onnx
        if onnx is None:
            print('onnx package missing; installing...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'onnx'])
            import importlib
            onnx = importlib.import_module('onnx')
        
        from onnx import helper, TensorProto

        model = onnx.load(onnx_path)
        graph = model.graph

        nodes = list(graph.node)
        new_nodes = []
        
        patched_count = 0
        for node in nodes:
            if node.op_type == 'Split':
                # If NumOutputs is 1, it's often a no-op or resizing artifact from tracing.
                if len(node.output) == 1:
                    # Replace Split with Identity to bypass the size check
                    identity_node = helper.make_node(
                        'Identity',
                        inputs=[node.input[0]],
                        outputs=[node.output[0]],
                        name=node.name + '_identity'
                    )
                    new_nodes.append(identity_node)
                    print(f"Patched Split node '{node.name}': Replaced with Identity (NumOutputs=1)")
                    patched_count += 1
                    continue
            
            new_nodes.append(node)

        if patched_count > 0:
            graph.ClearField('node')
            graph.node.extend(new_nodes)
            onnx.save(model, onnx_path)
            print(f'Patched ONNX: handled {patched_count} Split nodes.')
        else:
            print('No Split nodes required patching.')

    try:
        make_split_dynamic(args.onnx_path)
    except Exception as e:
        print('Warning: failed to post-process ONNX for dynamic splits:', e)
        import traceback
        traceback.print_exc()

    # ── Step 2: onnx-simplifier ──────────────────────────────────────────────
    def simplify_onnx(onnx_path):
        try:
            import onnxsim
        except ImportError:
            print('onnxsim not installed — skipping simplification (pip install onnxsim)')
            return
        print('Running onnx-simplifier …')
        model_proto = onnx.load(onnx_path)
        # onnxsim.simplify returns (simplified_model, check_ok)
        try:
            simplified, check_ok = onnxsim.simplify(model_proto)
        except Exception as e:
            print(f'  onnxsim failed: {e}')
            return
        if check_ok:
            onnx.save(simplified, onnx_path)
            orig_nodes = len(model_proto.graph.node)
            simp_nodes = len(simplified.graph.node)
            print(f'  Simplified: {orig_nodes} → {simp_nodes} nodes (removed {orig_nodes - simp_nodes})')
        else:
            print('  onnxsim check failed — keeping original graph')

    try:
        simplify_onnx(args.onnx_path)
    except Exception as e:
        print('Warning: onnxsim step failed:', e)
        import traceback
        traceback.print_exc()

    # ── Step 3: OnnxRuntime full-graph optimization ──────────────────────────
    def ort_optimize(onnx_path):
        try:
            import onnxruntime as ort
        except ImportError:
            print('onnxruntime not installed — skipping ORT optimization')
            return
        print('Running OnnxRuntime ORT_ENABLE_ALL optimization …')
        opt_path = onnx_path.replace('.onnx', '_opt.onnx')
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = opt_path
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            _ = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        except Exception:
            # Fall back to CPU-only if CUDA provider unavailable
            providers = ['CPUExecutionProvider']
            _ = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        if os.path.exists(opt_path):
            orig_size = os.path.getsize(onnx_path) / 1e6
            opt_size  = os.path.getsize(opt_path)  / 1e6
            print(f'  ORT-optimised model saved → {opt_path}  ({orig_size:.1f} MB → {opt_size:.1f} MB)')
            # Replace the main output path with the ORT-optimised version
            import shutil
            shutil.move(opt_path, onnx_path)
            print(f'  Replaced {onnx_path} with ORT-optimised model')
        else:
            print('  ORT did not write an optimised file (model already optimal or provider limitation)')

    try:
        ort_optimize(args.onnx_path)
    except Exception as e:
        print('Warning: ORT optimization step failed:', e)
        import traceback
        traceback.print_exc()

    # ── Step 4: Embed class names as ONNX metadata ───────────────────────────
    # Done after all graph-level optimisations so the metadata is never stripped.
    def embed_class_names(onnx_path, cfg):
        """Write obj_classes and rel_classes as JSON strings in the ONNX custom_metadata_map."""
        import json
        from sgg_benchmark.data import get_dataset_statistics

        print('Embedding class names into ONNX metadata ...')
        try:
            stats = get_dataset_statistics(cfg)
        except Exception as e:
            print(f'  Warning: failed to load dataset statistics: {e}')
            return

        obj_classes = stats.get('obj_classes', [])
        rel_classes = stats.get('rel_classes', [])

        # obj_classes may be a list or a dict keyed by index.
        # In both cases, build a flat list, dropping the background entry (index 0).
        if isinstance(obj_classes, dict):
            obj_list = [obj_classes[i] for i in sorted(obj_classes) if i != 0]
        else:
            # list: index 0 is background — skip it
            obj_list = list(obj_classes[1:]) if obj_classes else []

        if isinstance(rel_classes, dict):
            rel_list = [rel_classes[i] for i in sorted(rel_classes)]
        else:
            rel_list = list(rel_classes)

        model_proto = onnx.load(onnx_path)

        def _upsert_meta(proto, key, value):
            for entry in proto.metadata_props:
                if entry.key == key:
                    entry.value = value
                    return
            entry = proto.metadata_props.add()
            entry.key = key
            entry.value = value

        _upsert_meta(model_proto, 'obj_classes', json.dumps(obj_list))
        _upsert_meta(model_proto, 'rel_classes', json.dumps(rel_list))

        onnx.save(model_proto, onnx_path)
        print(f'  Embedded {len(obj_list)} object classes and {len(rel_list)} relation classes.')

    try:
        embed_class_names(args.onnx_path, cfg)
    except Exception as e:
        print('Warning: failed to embed class names into ONNX metadata:', e)
        import traceback
        traceback.print_exc()

    # ── Step 5: Final validation ─────────────────────────────────────────────
    def validate_onnx(onnx_path, sample_img):
        print('Validating final ONNX model …')
        import onnxruntime as ort
        size_mb = os.path.getsize(onnx_path) / 1e6

        # Count nodes with onnx proto (may fail for ORT-fused ops — use try/except)
        nodes = '(unknown)'
        try:
            model_proto = onnx.load(onnx_path)
            nodes = len(model_proto.graph.node)
            # Standard checker: skip for ORT-optimised models which may contain
            # ORT-specific ops (e.g. SimplifiedLayerNormalization) unknown to onnx.checker
            try:
                onnx.checker.check_model(model_proto)
                print(f'  ✓ ONNX standard checker passed')
            except Exception as ce:
                print(f'  ⚠ ONNX standard checker skipped (ORT-specific ops present): {ce}')
        except Exception:
            pass

        # Functional validation via ORT inference
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess = ort.InferenceSession(onnx_path, providers=providers)
            inp_name = sess.get_inputs()[0].name
            feed = {inp_name: sample_img.cpu().numpy()}
            outs = sess.run(None, feed)
            print(f'  ✓ ORT inference succeeded — {len(outs)} output tensor(s)')
            for i, o in enumerate(outs):
                print(f'      output[{i}]: shape={list(o.shape)} dtype={o.dtype}')
        except Exception as e:
            print(f'  ⚠ ORT inference check failed: {e}')

        inputs_info  = [(i.name, i.shape) for i in sess.get_inputs()]
        outputs_info = [(o.name, o.shape) for o in sess.get_outputs()]
        meta = sess.get_modelmeta().custom_metadata_map
        obj_n = len(__import__('json').loads(meta['obj_classes'])) if 'obj_classes' in meta else 0
        rel_n = len(__import__('json').loads(meta['rel_classes'])) if 'rel_classes' in meta else 0
        meta_status = (f'{obj_n} obj classes, {rel_n} rel classes' if obj_n
                       else '⚠ no embedded class names (--config will be required at inference)')

        print(f'  Size     : {size_mb:.1f} MB')
        print(f'  Nodes    : {nodes}')
        print(f'  Inputs   : {inputs_info}')
        print(f'  Outputs  : {outputs_info}')
        print(f'  Metadata : {meta_status}')

    try:
        validate_onnx(args.onnx_path, sample_img)
    except Exception as e:
        print('Warning: final validation failed:', e)

    print(f"\nDone — optimised model saved to: {args.onnx_path}")


if __name__ == '__main__':
    main()
