# Training a Custom-Domain SGG Model with a plain YOLO12 Backbone

Companion guide to **[train_custom_model.md](train_custom_model.md)** (the
YOLO-World path). Same running example — construction-site safety with
`person / helmet / safety-vest` and a `wearing` relation — but built on the
**closed-vocabulary YOLO12m** detector instead of YOLO-World.

**When to prefer YOLO12 over YOLO-World:**

- your vocabulary is fixed and small (you will fine-tune anyway);
- you want a much lighter detector (**~68 vs ~274 GFLOPs** at 640 for
  yolo12m vs yolov8x-worldv2, and a 41 MB vs 449 MB backbone file);
- you want the exact same code path as the PSG REACT++ release, with no
  `txt_feats` / CLIP vocabulary handling anywhere in the pipeline.

Stick to YOLO-World if you need open-vocabulary bootstrapping (zero-shot on a
new vocabulary) or expect the class list to change without retraining.

Everything below was executed and measured on this repository (RTX 3080 Ti):

| Stage | Metric | Value |
|---|---|---|
| Backbone fine-tune (ultralytics, val) | detection mAP50 | **0.947** |
| SGG evaluation (test split, sgdet) | detection mAP@0.5 | **0.914** |
| SGG evaluation (test split, sgdet) | R@20 / R@100 (`wearing`) | **0.927 / 0.968** |
| SGG training | wall clock | 58 min (20 epochs, 174 s/epoch) |

> SGG figures are measured on the **regenerated** dataset (9,761 `wearing`
> edges — one helmet **and** one vest per person, see the main guide §3). On
> the original single-edge synthesis (6,641 edges) this model scored test
> R@100 0.9680.

Pipeline overview:

```
 datasets/SAFETY/ppe-detection ──► build_safety_coco.py ──► datasets/SAFETY/coco_format   (SGG training data)
        (YOLO bbox only)         (main guide §2–3)     └──► datasets/SAFETY/safety-yolo  (backbone fine-tune)
                                                                      │
                                                                      ▼
                                    prepare_safety_yolo12_backbone.py --size yolo12m
                                                                      │  checkpoints/BACKBONES/SAFETY/yolo12m_backbone.pt
                                                                      ▼
                                    relation_train_net_hydra.py --config-name SAFETY/REACT_yolo12
                                                                      │  checkpoints/SAFETY/react_yolo12/
                                                                      ├──► relation_eval_hydra.py --run-dir ...
                                                                      └──► export_onnx.py ──► react_yolo12.onnx
```

---

## 1. Shared setup — same as the YOLO-World guide

Schema design, data collection and dataset conversion are **identical** to the
main guide (§1–§3): the converter produces both the COCO-SG jsons in
`datasets/SAFETY/coco_format` and the cleaned YOLO dataset in
`datasets/SAFETY/safety-yolo`. Complete those steps first.

## 2. Backbone fine-tuning — `scripts/prepare_safety_yolo12_backbone.py`

```powershell
uv run python scripts/prepare_safety_yolo12_backbone.py --size yolo12m --epochs 30 --batch 12
```

- starts from official COCO-pretrained `yolo12m.pt` (auto-downloaded to the
  repo root) and fine-tunes on `datasets/SAFETY/safety-yolo/data.yaml` (the
  default `--data`; repo-root relative, so run from the repo root);
- saves the best checkpoint to `checkpoints/BACKBONES/SAFETY/yolo12m_backbone.pt`;
- measured run: 30 epochs in ~37 min, **best mAP50 0.947** (epoch 29).

Because the vocabulary is closed, there is **no** `set_classes` / CLIP prompt
step and no `txt_feats` to keep track of — this is the main simplification vs
`prepare_safety_world_backbone.py` (main guide §4). Class names come from the
`data.yaml` (`person / helmet / safety-vest`) and must simply match the COCO-SG
category order.

Backbone variants and the matching config `yolo.out_channels`:

| Variant | `yolo.size` | `out_channels` (P3/P4/P5) |
|---|---|---|
| nano | `yolo12n` | `[64, 128, 256]` |
| small | `yolo12s` | `[128, 256, 512]` |
| medium | `yolo12m` | `[256, 512, 512]` |

## 3. Experiment config — `configs/hydra/SAFETY/REACT_yolo12.yaml`

Copy of `REACT_world.yaml` with four meaningful differences:

```yaml
output_dir: "./checkpoints/SAFETY/react_yolo12"
model:
  pretrained_detector_ckpt: "checkpoints/BACKBONES/SAFETY/yolo12m_backbone.pt"
  backbone:
    type: "yolo"                  # plain YOLO registry key (was "yoloworld")
  yolo:
    size: "yolo12m"
    out_channels: [256, 512, 512] # yolo12m P3/P4/P5
```

Everything else (DAMP feature extractor, `REACTPlusPlusPredictor` relation
head, solver, loss) is identical to the world config.

## 4. Training

```powershell
uv run python tools/relation_train_net_hydra.py `
    --config-name SAFETY/REACT_yolo12 --task sgdet
```

- defaults to 20 epochs; backbone frozen (`model.backbone.freeze: true`), only
  the relation head trains;
- measured: 174 s/epoch on 3.4k images, 58 min total; best epoch was 19
  (validation mR@k 0.969).

## 5. Evaluation

```powershell
uv run python tools/relation_eval_hydra.py `
    --run-dir checkpoints/SAFETY/react_yolo12 --task sgdet
```

Measured test-split output:

```
Detection evaluation mAp=0.9139
SGG eval: R @ 20: 0.9269;  R @ 50: 0.9620;  R @ 100: 0.9675  for mode=sgdet.
(wearing:0.9675)
```

## 6. ONNX export & deployment

Same procedure as the main guide §8:

```powershell
uv run python tools/export_onnx.py `
    --run-dir checkpoints/SAFETY/react_yolo12 `
    --image datasets/SAFETY/coco_format/test/test_000030.jpg `
    --onnx-path checkpoints/SAFETY/react_yolo12/react_yolo12.onnx

# smoke test (works for any REACT ONNX export despite the script name)
uv run python tools/test_react_world_onnx.py `
    --onnx checkpoints/SAFETY/react_yolo12/react_yolo12.onnx --source <image>
```

The output contract is identical to the world model (input `1×3×640×640` RGB
letterboxed `/255`; `boxes [N, 6]` with label 1=person, 2=helmet, 3=safety-vest
in letterbox space; `rels [M, 5]` with `label` 1 = `wearing`). The
`txt_feats`-restore step in `export_onnx.py` is a no-op concern here — a plain
YOLO backbone has no text features, so the "Restored YOLO-World txt_feats"
message simply will not appear.

## 7. Violation detection at inference

Identical to the main guide §9: a `person` box with no outgoing `wearing` edge
to the equipment class of interest is the violation.

## 8. Measured comparison — YOLO-World vs YOLO12 (SAFETY test split)

| | REACT_world (yolov8x-worldv2) | REACT_yolo12 (yolo12m) |
|---|---|---|
| Detector compute (~640) | ~274 GFLOPs | ~68 GFLOPs |
| Backbone file | 449 MB | 41 MB |
| Relation GT evaluated on | regenerated (9,761 edges) | regenerated (9,761 edges) |
| Detection mAP@0.5 | 0.9079 | 0.9139 |
| R@100 (`wearing`) | 0.9799 | 0.9675 |
| Vocabulary | open (CLIP `txt_feats`) | fixed at train time |
| Export extras | `txt_feats` restore needed | none |

Both models were retrained on the same regenerated GT; on the original
single-edge GT (6,641 edges) they scored test R@100 0.9806 / 0.9680 — i.e.
the 47%-denser GT barely moved either model's recall.

Accuracy is close, with an edge for YOLO-World on relation recall; YOLO12
buys a ~4× lighter detector and a simpler deployment path.

## 9. Troubleshooting — delta vs the main guide

All pitfalls of the main guide §10 apply unchanged (category ids from 1, stale
`*.cache`, 1-based `pred_labels`, empty-detection fallbacks, HF 429s), plus
one YOLO12-specific entry:

| # | Symptom | Cause → fix |
|---|---|---|
| 1 | `ImportError: cannot import name 'feature_visualization'` on import | ultralytics ≥ 8.4 removed it from `ultralytics.utils.plotting`. Already fixed in-tree: the import in `sgg_benchmark/modeling/backbone/yolo.py` / `yoloe.py` falls back to the identical local copy in `backbone/utils.py`. |

## 10. Dataset license

Same as the main guide §11: **CC BY 4.0** (Roboflow / VincentGOURBIN) — keep
the attribution with your model cards.
