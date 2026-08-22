"""Fine-tune a plain YOLO12 backbone on the safety YOLO dataset (same data as
prepare_safety_world_backbone.py --mode fine-tune, but closed-vocabulary:
no CLIP vocabulary / set_classes step).

Usage:
  python scripts/prepare_safety_yolo12_backbone.py --size yolo12m --epochs 30
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="yolo12m", help="yolo12n / yolo12s / yolo12m / yolo12l")
    ap.add_argument("--data", default="datasets/SAFETY/safety-yolo/data.yaml")
    ap.add_argument("--out", default=None,
                    help="default: checkpoints/BACKBONES/SAFETY/<size>_backbone.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    out = Path(args.out) if args.out else Path(f"checkpoints/BACKBONES/SAFETY/{args.size}_backbone.pt")
    model = YOLO(f"{args.size}.pt")  # official COCO-pretrained weights (auto-download)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
    best = YOLO(model.trainer.best)
    out.parent.mkdir(parents=True, exist_ok=True)
    best.save(str(out))
    print(f"saved fine-tuned {args.size} (best={model.trainer.best}) -> {out}")


if __name__ == "__main__":
    main()
