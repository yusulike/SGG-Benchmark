"""Build a COCO-SG dataset for construction-site safety SGG.

Converts a YOLO-format detection dataset (class ids: 0=person, 1=helmet,
2=safety-vest) into the COCO-SG JSON format expected by the SGG-Benchmark
training pipeline, synthesizing "wearing" relations geometrically:

    helmet  -> person with the highest containment  ratio(helmet & person) / area(helmet)
    vest    -> person with the highest containment  ratio(vest   & person) / area(vest)

Each equipment object is assigned to at most one person (greedy, best score
first); a person with no assigned helmet/vest is a potential violation case.

Input layout (either):
    A) <src>/images/*.jpg + <src>/labels/*.txt          (single pool, random split)
    B) <src>/{train,val,test}/{images,labels}/...       (pre-split)

Output: datasets/SAFETY/coco_format/{train,val,test}/_annotations.coco.json
        with images copied alongside.

Usage:
    python process_data/build_safety_coco.py --src /path/to/yolo_dataset
    python process_data/build_safety_coco.py --src /path/to/yolo_dataset \
        --dst datasets/SAFETY/coco_format --helmet-contain 0.5 --vest-contain 0.6
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

# JSON category ids MUST start at 1: the dataloader reserves id 0 for
# "__background__" (COCO convention). The YOLO label ids fed into this script
# (0=person, 1=helmet, 2=safety-vest) are shifted by +1 on output.
CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "helmet"},
    {"id": 3, "name": "safety-vest"},
]
REL_CATEGORIES = [{"id": 1, "name": "wearing"}]

PERSON, HELMET, VEST = 0, 1, 2
TARGET_NAMES = {PERSON: "person", HELMET: "helmet", VEST: "safety-vest"}
# accepted --class-map target names (aliases included)
TARGET_ALIASES = {"person": PERSON, "helmet": HELMET, "vest": VEST, "safety-vest": VEST}


def parse_class_map(pairs):
    """['3:person', '0:helmet', '4:vest'] -> {3: 0, 0: 1, 4: 2} (src id -> target id)."""
    mapping = {}
    for p in pairs:
        src, name = p.split(":", 1)
        name = name.strip().lower()
        if name not in TARGET_ALIASES:
            raise ValueError(f"class-map target must be one of {sorted(TARGET_ALIASES)}, got '{name}'")
        mapping[int(src)] = TARGET_ALIASES[name]
    return mapping


def parse_yolo_labels(txt_path: Path, img_w: int, img_h: int, class_map=None):
    """YOLO txt (cls cx cy w h, normalized) -> boxes in absolute xywh.

    When `class_map` is given (src id -> target id), unmapped classes are
    dropped and kept classes are renumbered.
    """
    boxes = []
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if class_map is not None:
            if cls not in class_map:
                continue
            cls = class_map[cls]
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append({
            "cls": cls,
            "bbox": [
                max(0.0, (cx - w / 2) * img_w),
                max(0.0, (cy - h / 2) * img_h),
                min(w * img_w, img_w),
                min(h * img_h, img_h),
            ],
            "raw": (parts[1], parts[2], parts[3], parts[4]),
        })
    return boxes


def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def contain_ratio(inner, outer):
    """Area overlap relative to the inner box."""
    ia = inner[2] * inner[3]
    if ia <= 0:
        return 0.0
    return intersection_area(inner, outer) / ia


def synthesize_relations(boxes, helmet_min=0.5, vest_min=0.6, head_expand=0.05):
    """Assign each helmet/vest to the best-matching person.

    `head_expand` grows the person box upwards before matching helmets, since
    helmet boxes often sit at or slightly above the annotated person's head.
    Returns a list of (subject_idx, obj_idx) pairs into `boxes`.
    """
    persons = [i for i, b in enumerate(boxes) if b["cls"] == PERSON]
    if not persons:
        return []

    # candidate (score, equipment_idx, person_idx)
    candidates = []
    for i, b in enumerate(boxes):
        if b["cls"] not in (HELMET, VEST):
            continue
        thr = helmet_min if b["cls"] == HELMET else vest_min
        probe = b["bbox"]
        for p in persons:
            person_box = boxes[p]["bbox"]
            # person box expanded upwards for helmets
            target = person_box
            if b["cls"] == HELMET:
                target = [
                    person_box[0],
                    max(0.0, person_box[1] - head_expand * person_box[3]),
                    person_box[2],
                    person_box[3] * (1 + head_expand),
                ]
            score = contain_ratio(probe, target)
            if score >= thr:
                candidates.append((score, i, p))

    # greedy one-to-one assignment, best score first
    candidates.sort(reverse=True)
    used_persons, used_equip, pairs = set(), set(), []
    for score, i, p in candidates:
        if i in used_equip or p in used_persons:
            continue
        used_equip.add(i)
        used_persons.add(p)
        pairs.append((p, i))  # (person, equipment) -> subject, object
    return pairs


def emit_yolo(items_by_split, out_dir: Path, class_map):
    """Write a cleaned YOLO-format dataset (remapped classes, unmapped dropped)
    plus data.yaml — ready for ultralytics backbone fine-tuning."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split_names = []
    for split, items in items_by_split.items():
        if not items:
            continue
        split_names.append(split)
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        n_kept = 0
        for img_path, txt_path in items:
            shutil.copy2(img_path, out_dir / split / "images" / img_path.name)
            lines = []
            for line in txt_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if class_map is not None:
                    if cls not in class_map:
                        continue
                    cls = class_map[cls]
                lines.append(f"{cls} " + " ".join(parts[1:5]))
                n_kept += 1
            (out_dir / split / "labels" / (txt_path.stem + ".txt")).write_text("\n".join(lines))
        print(f"[emit-yolo:{split}] {len(items)} images, {n_kept} boxes -> {out_dir / split}")
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images" if "val" in split_names else "train/images",
        "test": "test/images" if "test" in split_names else None,
        "nc": 3,
        "names": ["person", "helmet", "safety-vest"],
    }
    (out_dir / "data.yaml").write_text(
        "\n".join(f"{k}: {v}" if v is not None else f"{k}:" for k, v in data_yaml.items())
    )
    print(f"[emit-yolo] data.yaml -> {out_dir / 'data.yaml'}")


def build_split(items, split_name, dst, args):
    """Write one split: copy images + emit COCO-SG json."""
    split_dir = dst / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    images, annotations, rel_annotations = [], [], []
    ann_id = 1
    rel_id = 1
    stats = {"images": 0, "objects": 0, "relations": 0}

    for img_path, txt_path in items:
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        img_id = len(images) + 1
        file_name = f"{split_name}_{img_id:06d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, split_dir / file_name)
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": img_w,
            "height": img_h,
        })

        boxes = parse_yolo_labels(txt_path, img_w, img_h, args.class_map)
        obj_ids = []
        for b in boxes:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": b["cls"] + 1,  # +1: id 0 is reserved for background
                "bbox": [round(v, 2) for v in b["bbox"]],
                "area": round(b["bbox"][2] * b["bbox"][3], 2),
                "iscrowd": 0,
            })
            obj_ids.append(ann_id)
            ann_id += 1

        for subj_idx, obj_idx in synthesize_relations(
            boxes, args.helmet_contain, args.vest_contain, args.head_expand
        ):
            rel_annotations.append({
                "id": rel_id,
                "image_id": img_id,
                "subject_id": obj_ids[subj_idx],
                "object_id": obj_ids[obj_idx],
                "predicate_id": 1,  # "wearing"; id 0 reserved for background
            })
            rel_id += 1

        stats["images"] += 1
        stats["objects"] += len(boxes)

    stats["relations"] = len(rel_annotations)
    coco = {
        "images": images,
        "annotations": annotations,
        "rel_annotations": rel_annotations,
        "categories": CATEGORIES,
        "rel_categories": REL_CATEGORIES,
    }
    out = split_dir / "_annotations.coco.json"
    out.write_text(json.dumps(coco, indent=1))
    print(f"[{split_name}] {stats['images']} images, {stats['objects']} objects, "
          f"{stats['relations']} relations -> {out}")
    return stats


def collect_items(src: Path):
    """Return {split: [(img, txt), ...]} from either input layout."""
    if (src / "train" / "images").is_dir() or (src / "valid" / "images").is_dir():
        splits = {}
        for split, alt in (("train", "train"), ("val", "valid"), ("test", "test")):
            img_dir = src / alt / "images"
            lbl_dir = src / alt / "labels"
            if img_dir.is_dir():
                splits[split] = pair_images_labels(img_dir, lbl_dir)
        return splits

    img_dir, lbl_dir = src / "images", src / "labels"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"Expected YOLO layout at {src}: images/ + labels/ or train/val/test subdirs"
        )
    return {"__pool__": pair_images_labels(img_dir, lbl_dir)}


def pair_images_labels(img_dir: Path, lbl_dir: Path):
    items = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        txt = lbl_dir / (img.stem + ".txt")
        if txt.exists():
            items.append((img, txt))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="YOLO-format dataset root")
    ap.add_argument("--dst", default="datasets/SAFETY/coco_format",
                    help="Output directory (default: datasets/SAFETY/coco_format)")
    ap.add_argument("--class-map", nargs="+", metavar="SRC:TARGET",
                    help="Map source YOLO class ids to person/helmet/vest "
                         "(e.g. 3:person 0:helmet 4:vest); unmapped classes are dropped. "
                         "Default assumes src ids 0=person 1=helmet 2=vest.")
    ap.add_argument("--emit-yolo", metavar="DIR", default=None,
                    help="Also write a cleaned YOLO dataset + data.yaml (remapped "
                         "classes) for ultralytics backbone fine-tuning")
    ap.add_argument("--helmet-contain", type=float, default=0.5,
                    help="Min containment ratio(helmet & person)/area(helmet) (default 0.5)")
    ap.add_argument("--vest-contain", type=float, default=0.6,
                    help="Min containment ratio(vest & person)/area(vest) (default 0.6)")
    ap.add_argument("--head-expand", type=float, default=0.05,
                    help="Upward person-box expansion fraction for helmet matching (default 0.05)")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio (default 0.1)")
    ap.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (default 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    args.class_map = parse_class_map(args.class_map) if args.class_map else None
    items_by_split = collect_items(src)

    if "__pool__" in items_by_split:
        pool = items_by_split.pop("__pool__")
        rng = random.Random(args.seed)
        rng.shuffle(pool)
        n_val = int(len(pool) * args.val_ratio)
        n_test = int(len(pool) * args.test_ratio)
        items_by_split["test"] = pool[:n_test]
        items_by_split["val"] = pool[n_test:n_test + n_val]
        items_by_split["train"] = pool[n_test + n_val:]

    if args.emit_yolo:
        emit_yolo(items_by_split, Path(args.emit_yolo), args.class_map)

    total = {"images": 0, "objects": 0, "relations": 0}
    for split in ("train", "val", "test"):
        if not items_by_split.get(split):
            print(f"[{split}] empty, skipping")
            continue
        s = build_split(items_by_split[split], split, dst, args)
        for k in total:
            total[k] += s[k]

    print(f"\nDone. Total: {total['images']} images, {total['objects']} objects, "
          f"{total['relations']} wearing-relations.")
    print(f"Next: python tools/relation_train_net_hydra.py --config-name SAFETY/REACT_world --task sgdet")


if __name__ == "__main__":
    main()
