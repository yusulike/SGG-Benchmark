# Training a Custom-Domain SGG Model — Complete Walkthrough

This guide documents a **verified end-to-end pipeline** for building a scene-graph
model on your own domain, using a YOLO-World open-vocabulary backbone. The running
example is **construction-site safety** (objects: `person`, `helmet`, `safety-vest`;
relation: `wearing`) but every step generalizes to any small closed vocabulary.
A closed-vocabulary YOLO12 variant of this guide (with a measured comparison against
this YOLO-World path) is available in [train_custom_model_yolo12.md](train_custom_model_yolo12.md).

Everything below was executed and measured on this repository; final results of the
example run:

| Stage | Metric | Value |
|---|---|---|
| Backbone fine-tune (ultralytics, val) | detection mAP50 | **0.952** |
| SGG evaluation (test split, sgdet) | detection mAP@0.5 | **0.908** |
| SGG evaluation (test split, sgdet) | R@20 / R@100 (`wearing`) | **0.935 / 0.980** |

> SGG figures are measured on the **regenerated** dataset (9,761 `wearing`
> edges — one helmet **and** one vest per person, §3). On the original
> single-edge synthesis (6,641 edges) the same pipeline scored test R@100
> 0.9806.

> Commands use `uv run` from the repo root — the root `pyproject.toml` now
> declares the full environment (torch cu121, ultralytics, hydra, ...), so a
> one-time `uv sync` is all the setup you need (see INSTALL.md).

Pipeline overview:

```
 public YOLO-format dataset ──► build_safety_coco.py ──► COCO-SG JSON (datasets/SAFETY/)
        (bbox only)                (relation            │
                                    synthesis)          └──► cleaned YOLO + data.yaml (backbone fine-tune)
                                                              │
                                                              ▼
                                        prepare_safety_world_backbone.py --mode fine-tune
                                                              │  checkpoints/BACKBONES/SAFETY/worldv2_backbone.pt
                                                              ▼
                                        relation_train_net_hydra.py --config-name SAFETY/REACT_world
                                                              │  checkpoints/SAFETY/react_world/
                                                              ├──► relation_eval_hydra.py --run-dir ...
                                                              └──► export_onnx.py ──► react_world.onnx
```

---

## 1. Schema design

| Slot | Values | Notes |
|---|---|---|
| Objects (3) | `person`, `helmet`, `safety-vest` | **JSON category ids start at 1** — the dataloader reserves id 0 for `__background__` |
| Relations (1) | `wearing` | triplets are always `(person, wearing, helmet|vest)` |
| Violations | inferred | a `person` with **no outgoing `wearing` edge** = missing helmet/vest (see §9) |

With a single predicate the task reduces to person↔equipment *association*: the model
learns which helmet/vest belongs to which person, and a missing edge is the violation.

## 2. Data collection

You need bounding boxes only — relations are synthesized geometrically in the next
step (§3), so **any YOLO-format detection dataset works**.

For the safety domain, suitable public sets were found by searching the HuggingFace
hub API:

```bash
curl -s "https://huggingface.co/api/datasets?search=PPE%20detection" | python -c \
    "import json,sys; [print(d['id']) for d in json.load(sys.stdin)]"
```

We used **[VincentGOURBIN/ppe-detection](https://huggingface.co/datasets/VincentGOURBIN/ppe-detection)**
(CC BY 4.0, ~3.7k images, Roboflow YOLO export, classes `helmet / no-helmet / no-vest /
person / vest`). Download with `snapshot_download`; two practical tips:

- Repos with thousands of small files can hit HTTP 429 on the **xet** endpoint —
  set `HF_HUB_DISABLE_XET=1` and retry in a loop (resumes automatically).
- Export `HF_TOKEN` (or read it from the registry on Windows) for higher rate limits.

```python
import os, time
from huggingface_hub import snapshot_download

for attempt in range(10):
    try:
        p = snapshot_download(
            repo_id="VincentGOURBIN/ppe-detection",
            repo_type="dataset",
            local_dir="datasets/SAFETY/ppe-detection",
            max_workers=4,
        )
        break
    except Exception as e:
        print("retry:", e); time.sleep(60)
```

(If you annotate your own images instead, any labeling tool that exports YOLO
format works; the original author's [SGG-Annotate](https://github.com/Maelic/SGG-Annotate)
can also emit relation labels directly.)

## 3. Dataset conversion — `process_data/build_safety_coco.py`

The converter maps **source class ids → the 3 target classes** (unmapped classes are
dropped — e.g. the negative markers `no-helmet`/`no-vest`), synthesizes `wearing`
relations, and optionally emits a cleaned YOLO dataset for backbone fine-tuning:

```powershell
uv run python process_data/build_safety_coco.py `
    --src datasets/SAFETY/ppe-detection `
    --class-map 3:person 0:helmet 4:vest `
    --emit-yolo datasets/SAFETY/safety-yolo
```

Relation synthesis rules (tunable via `--helmet-contain 0.5 --vest-contain 0.6
--head-expand 0.05`):

- **helmet → person**: the person box is expanded upward by `head_expand`; the helmet
  is assigned to the person with the highest containment
  `area(helmet ∩ person) / area(helmet)` above the threshold;
- **vest → person**: same containment rule without expansion;
- each equipment box is assigned to **at most one** person, and each person to
  at most **one helmet and one vest** (greedy, best containment first) — a person
  wearing both keeps both `wearing` edges, so violation inference (§9) can check
  helmet and vest compliance independently.

Example-run output:

```
[train] 3447 images, 17249 objects, 9045 relations
[val]    164 images,   867 objects,  452 relations
[test]   101 images,   523 objects,  264 relations
```

Outputs land in `datasets/SAFETY/coco_format/{train,val,test}/_annotations.coco.json`
(COCO-SG schema: `images` / `annotations` / `rel_annotations` / `categories` /
`rel_categories`) — the layout the training config points at.

## 4. Backbone preparation — `scripts/prepare_safety_world_backbone.py`

Three modes:

```powershell
# a) probe: verify feature channels for yolo.out_channels in the config (no weights)
uv run python scripts/prepare_safety_world_backbone.py --mode probe
#    -> yolov8x-worldv2 feature layers {15,18,21} = [320, 640, 640]

# b) zero-shot: official weights re-prompted with your vocabulary (bootstrap only)
uv run python scripts/prepare_safety_world_backbone.py --mode zero-shot

# c) fine-tune (recommended): ultralytics training on the cleaned YOLO data
uv run python scripts/prepare_safety_world_backbone.py --mode fine-tune `
    --data datasets/SAFETY/safety-yolo/data.yaml --epochs 10 --batch 12
```

The fine-tuned backbone is saved to
`checkpoints/BACKBONES/SAFETY/worldv2_backbone.pt` (it also stores the CLIP
`txt_feats` of your vocabulary — see troubleshooting §10.4). Fine-tune took ~49 min
(10 epochs, RTX 3080 Ti) and reached **mAP50 0.952** (person 0.936 / helmet 0.949 /
safety-vest 0.970).

Prompt wording matters for the open-vocabulary head — the default vocabulary uses
`"high-visibility safety vest"` because it matches CLIP text embeddings better than
the bare class name.

## 5. Experiment config — `configs/hydra/SAFETY/REACT_world.yaml`

Key differences from the PSG config (copy `configs/hydra/PSG/REACT++.yaml` and edit):

```yaml
output_dir: "./checkpoints/SAFETY/react_world"
datasets:
  name: "SAFETY"
  type: "coco"
  data_dir: "datasets/SAFETY/coco_format"
model:
  pretrained_detector_ckpt: "checkpoints/BACKBONES/SAFETY/worldv2_backbone.pt"
  text_embedding: "clip"          # no GloVe download needed; matches YOLO-World
  backbone:
    type: "yoloworld"             # registry key for the WorldModel backbone
  yolo:
    size: "yolov8x-worldv2"
    out_channels: [320, 640, 640] # P3/P4/P5 C2fAttn outputs — verify with --mode probe
  roi_box_head:
    num_classes: 4                # 3 objects + background (auto-corrected from dataset)
  roi_relation_head:
    num_classes: 1                # "wearing" (auto-corrected from dataset)
    embed_dim: 512                # CLIP ViT-B/32 dim
```

## 6. Training

```powershell
uv run python tools/relation_train_net_hydra.py `
    --config-name SAFETY/REACT_world --task sgdet solver.max_epoch=10 solver.ims_per_batch=4
```

- ~3 min/epoch on 3.4k images (RTX 3080 Ti); backbone is frozen
  (`model.backbone.freeze: true`), only the relation head trains.
- The trainer auto-extracts class counts from the dataset and writes
  `config.yml` + `model_epoch_*.pth` into `output_dir`.

## 7. Evaluation

```powershell
uv run python tools/relation_eval_hydra.py `
    --run-dir checkpoints/SAFETY/react_world --task sgdet
```

Auto-discovers the newest `model_epoch_*.pth` and the config in the run dir (or pass
`--checkpoint`/`--config-file` explicitly). Example output:

```
Detection evaluation mAp=0.9079
SGG eval: R @ 20: 0.9345;  R @ 50: 0.9685;  R @ 100: 0.9799  for mode=sgdet.
(wearing:0.9799)
```

## 8. ONNX export & deployment

```powershell
# export (restores YOLO-World txt_feats from the backbone file automatically)
uv run python tools/export_onnx.py `
    --run-dir checkpoints/SAFETY/react_world `
    --image datasets/SAFETY/coco_format/test/test_000030.jpg `
    --onnx-path checkpoints/SAFETY/react_world/react_world.onnx

# smoke test
uv run python tools/test_react_world_onnx.py `
    --onnx checkpoints/SAFETY/react_world/react_world.onnx --source <image>
```

Output contract (input: `1×3×640×640` RGB letterboxed, `/255`):

- `boxes [N, 6]` = `(x1, y1, x2, y2, label, score)` — label **1=person, 2=helmet,
  3=safety-vest**; coordinates in the 640×640 letterbox space (reverse the letterbox
  for original-image coordinates).
- `rels [M, 5]` = `(subj_idx, obj_idx, label, triplet_score, rel_score)` — object
  indices point into `boxes`; `label` is 1 (`wearing`).

> **Calibration note:** absolute `rel_score` values stay low (~1e-3) even when
> ranking is excellent — filter relations by **relative rank / triplet_score
> ordering**, not by an absolute threshold like 0.5.

## 9. Violation detection at inference

```python
# boxes: (N,6), rels: (M,5) from the ONNX model
worn_by = {int(r[1]) for r in rels if int(r[2]) == 1}        # equipment objects worn
helmet_wearers = {int(r[0]) for r in rels if int(r[2]) == 1
                  and int(boxes[int(r[1])][4]) == 2}          # persons wearing a helmet
violators = [i for i in range(len(boxes))
             if int(boxes[i][4]) == 1 and i not in helmet_wearers]
```

## 10. Troubleshooting — pitfalls we actually hit

| # | Symptom | Cause → fix |
|---|---|---|
| 1 | Person class vanishes from statistics; `nc` off by one | JSON `categories` used id 0, which the loader overwrites with `__background__`. **Category ids must start at 1** (the converter does this). |
| 2 | Metrics don't change after rebuilding the dataset | Stale `SAFETY_statistics.cache` in `output_dir` — delete `*.cache` under the run dir whenever the data changes. |
| 3 | Perfect boxes but det mAP = 0.000, AR ≈ 0 | `pred_labels` emitted 0-based while the evaluator maps 1-based ids → every class shifted. Emit `pred_labels = yolo_label + 1`. |
| 4 | Exported ONNX: all class scores ≈ 0 | YOLO-World `txt_feats` is a plain attribute, **not** in state dicts; exporting from a full SGG checkpoint left it at random init. `export_onnx.py` now restores it from the backbone file — keep that file next to your run. |
| 5 | Crash on images with zero detections (`IndexError` in the sampler / `torch.cat` on `feat_idx`) | The sampler's dummy pair indexes proposal 0. Backbones now emit one background box (score 0) for empty images, and the NMS shim normalizes `keepi` shapes. |
| 6 | HF download dies with HTTP 429 on `xet-read-token` | `HF_HUB_DISABLE_XET=1` + authenticated token + retry loop (§2). |
| 7 | `uv run` cannot find `torch` | The repo-root `pyproject.toml` used to declare only the serving-app deps, so uv resolved an env without torch. **Fixed:** it now declares the full SGG dependency set (torch pinned to the cu121 wheel index) — run `uv sync` once and use `uv run python ...`. |

## 11. Dataset license

The example dataset is redistributed under **CC BY 4.0** (Roboflow / VincentGOURBIN).
Check the license of any dataset you download before commercial deployment, and keep
the attribution with your model cards.
