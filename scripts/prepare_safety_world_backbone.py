"""Prepare the YOLO-World v2 backbone for the construction-safety SGG project.

The SGG pipeline (configs/hydra/SAFETY/REACT_world.yaml) loads a YOLO-World
checkpoint whose vocabulary must match the safety object classes, because the
backbone forward pass feeds per-class CLIP text features (txt_feats) into the
WorldDetect head. Three modes:

  1) probe       : no weights needed — build the architecture from the yaml and
                   print the feature-map channels at the layers used by the
                   relation head (layers 15/18/21). Use this to verify
                   `yolo.out_channels` in the config.
  2) zero-shot   : download official yolov8x-worldv2.pt, re-prompt it with the
                   safety vocabulary and save. Good enough to bootstrap the
                   relation-head training; detection will be weaker on
                   construction imagery.
  3) fine-tune   : fine-tune yolov8x-worldv2 on your YOLO-format detection data
                   (same data used for build_safety_coco.py). Recommended once
                   you have a few thousand labelled boxes.

Usage:
  python scripts/prepare_safety_world_backbone.py --probe
  python scripts/prepare_safety_world_backbone.py --mode zero-shot
  python scripts/prepare_safety_world_backbone.py --mode fine-tune \
      --data /path/to/yolo_dataset/data.yaml --epochs 30
"""

import argparse
from pathlib import Path

# Prompt wording matters for the open-vocabulary head: "high-visibility vest"
# matches CLIP text embeddings better than the bare class name.
DEFAULT_CLASSES = ["person", "helmet", "high-visibility safety vest"]


def probe(size: str):
    import torch
    from ultralytics.nn.modules import C2fAttn
    from ultralytics.nn.modules.head import WorldDetect
    from ultralytics.nn.tasks import WorldModel

    model = WorldModel(f"{size}.yaml", nc=len(DEFAULT_CLASSES))
    model.eval()
    x = torch.zeros(1, 3, 640, 640)
    # zero text features (B, nc, embed_dim) — C2fAttn/WorldDetect need them
    txt_feats = torch.zeros(1, len(DEFAULT_CLASSES), 512)
    y = []
    feat_layers = {15, 18, 21}
    channels = {}
    with torch.no_grad():
        for m in model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, txt_feats)
            else:
                x = m(x)
            y.append(x if m.i in model.save else None)
            if m.i in feat_layers:
                channels[m.i] = tuple(x.shape)

    print(f"\n{size} module map:")
    for m in model.model:
        extra = f"  <-- feature layer, shape {channels[m.i]}" if m.i in channels else ""
        print(f"  {m.i:2d} {m.type:45s}{extra}")

    dims = [channels[i][1] for i in sorted(channels)]
    print(f"\nyolo.out_channels for configs/hydra/SAFETY/REACT_world.yaml: {dims}")
    warn = [i for i in feat_layers if i not in channels]
    if warn:
        print(f"WARNING: expected feature layers {warn} produced no output — "
              f"check yoloworld.py forward (extract set {{15, 18, 21}})")


def zero_shot(size: str, classes, out: Path):
    from ultralytics import YOLO

    model = YOLO(f"{size}.pt")  # auto-downloads official weights
    model.set_classes(classes)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"saved zero-shot {size} with vocabulary {classes} -> {out}")


def fine_tune(size: str, classes, out: Path, data: str, epochs: int, batch: int, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(f"{size}.pt")
    model.train(data=data, epochs=epochs, batch=batch, imgsz=imgsz)
    trained = YOLO(model.trainer.best)  # reload: best.pt carries txt_feats for the trained vocab
    trained.set_classes(classes)
    out.parent.mkdir(parents=True, exist_ok=True)
    trained.save(str(out))
    print(f"saved fine-tuned {size} (best={model.trainer.best}) -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["probe", "zero-shot", "fine-tune"], default="probe")
    ap.add_argument("--size", default="yolov8x-worldv2")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="Detection vocabulary (CLIP prompts)")
    ap.add_argument("--out", default="checkpoints/BACKBONES/SAFETY/worldv2_backbone.pt")
    ap.add_argument("--data", help="ultralytics data.yaml (fine-tune mode)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    out = Path(args.out)
    if args.mode == "probe":
        probe(args.size)
    elif args.mode == "zero-shot":
        zero_shot(args.size, args.classes, out)
    else:
        if not args.data:
            ap.error("--data (ultralytics data.yaml) is required for fine-tune")
        fine_tune(args.size, args.classes, out, args.data, args.epochs, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
