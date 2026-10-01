# Scene Graph Benchmark in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-2.2.1-%237732a8)
[![arXiv](https://img.shields.io/badge/arXiv-2405.16116-b31b1b.svg)](https://arxiv.org/abs/2603.06386)

## Custom-Domain SGG — Interactive Pipeline (safety / YOLO12)

Train a scene-graph model on your own domain (running example: construction-site
safety, `person / helmet / safety-vest` + `wearing`, YOLO12m backbone) with a
single interactive runner.

### From a fresh machine (only git required)

`uv` is the sole prerequisite — it manages Python itself, so no interpreter
needs to be pre-installed:

```bash
# 1) install uv
curl -LsSf https://astral.sh/uv/install.sh | sh            # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2) clone
git clone https://github.com/yusulike/SGG-Benchmark.git && cd SGG-Benchmark

# 3) build the training env — .venv-sgg (Python 3.12 + torch + requirements + CLIP + onnx)
uv run --no-project scripts/safety_pipeline.py --stage setup

# 4) run the whole pipeline — auto-switches into .venv-sgg once setup is done
uv run --no-project scripts/safety_pipeline.py --stage all --yes
```

- `--stage setup` asks for the torch wheel index: keep the default
  (`.../cu128`) on recent NVIDIA GPUs, answer `https://download.pytorch.org/whl/cpu`
  on CPU-only machines.
- Datasets and checkpoints are gitignored — a fresh clone contains code only;
  `--stage all` rebuilds everything (backbone ~35 min + SGG training ~40 min on
  a desktop GPU; training is impractical on CPU).
- First runs additionally download the HF dataset (~120 MB), `yolo12m.pt`
  COCO-pretrained weights and CLIP ViT-B/32 — all automatic.

### Day-to-day usage

```bash
# interactive stage menu: status, smart resume, per-stage prompts
uv run --no-project scripts/safety_pipeline.py
# or invoke the repo venv directly (plain `python` also works after activating it)
.venv-sgg/Scripts/python.exe scripts/safety_pipeline.py     # Windows
.venv-sgg/bin/python scripts/safety_pipeline.py             # Linux / macOS
# hands-free end-to-end (skips stages whose outputs already exist; --force to redo)
<python> scripts/safety_pipeline.py --stage all --yes
# single stage with overrides
<python> scripts/safety_pipeline.py --stage backbone --size yolo12m --epochs 30 --yes
<python> scripts/safety_pipeline.py --stage train --train-epochs 20 --ims 8
```

Stages: venv setup (uv) → dataset download (HuggingFace) → COCO-SG conversion →
backbone fine-tune → SGG training → evaluation → ONNX export + smoke test.
Full walkthrough, metrics and troubleshooting: [train_custom_model.md](train_custom_model.md).

## [Under Review] Code for the paper [REACT++: Efficient Cross-Attention for Real-Time Scene Graph Generation](https://arxiv.org/abs/2603.06386)

## [BMVC 2025] Code for the paper [REACT: Real-time Efficiency and Accuracy Compromise for Tradeoffs in Scene Graph Generation](https://arxiv.org/abs/2405.16116)

Previous work (PE-NET model) | Our REACT model for Real-Time SGG
:-: | :-:
<video src='https://github.com/user-attachments/assets/1e580ecc-6a31-409c-82b5-4488aadaf815' width=480/> | <video src='https://github.com/user-attachments/assets/6dfc22de-176a-4d50-9e3a-e91d8df76777' width=480/>

REACT++ versus previous work | The REACT++ family
:-: | :-:
<img alt="latency_vs_f1_psg" src="https://github.com/user-attachments/assets/81b27eaa-5e82-4401-a942-edd2e6c75422" width="480"/> | <img alt="react_pp_onnx_tradeoff" src="https://github.com/user-attachments/assets/8c7fd3ad-3750-4702-8b1b-9b97f6831108" width="480"/>

## Very Quick Start 🚀

If you don't want to install the codebase, we provide a minimal running example with ONNX Runtime under [demo/standalone_onnx_demo.py](demo/standalone_onnx_demo.py) that you can run with a single command after downloading an onnx model from the [MODEL_ZOO.md](docs/MODEL_ZOO.md):

```bash
# You need to have CUDA and cudnn installed for GPU inference with the onnxruntime-gpu package.
pip install onnxruntime-gpu opencv-python numpy
python demo/standalone_onnx_demo.py \
    --onnx checkpoints/PSG/react++_yolo12m/model.onnx \
    --rel_conf 0.05 --box_conf 0.4
```

## Quick Start 🚀

1. Install
```bash
# with uv (https://docs.astral.sh/uv/): creates .venv with all dependencies, GPU torch included
uv sync
```
See [INSTALL.md](docs/INSTALL.md) for details (including the legacy `scripts/install_uv.sh` installer).

2. Pick a model from [MODEL_ZOO.md](docs/MODEL_ZOO.md) and download it using 🤗 huggingface:
```bash
# Example: REACT++ PSG YOLOv12m (best accuracy/speed trade-off)
hf download maelic/REACTPlusPlus_PSG yolo12m/react_pp_yolo12m.onnx \
    --repo-type model --local-dir checkpoints/PSG/react++_yolo12m
```

3. Run inference with a webcam demo
```bash
python demo/webcam_demo_onnx.py \
    --onnx  checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx \
    --rel_conf 0.05 --box_conf 0.4
```

Check the [demo](demo/) folder for more demos and details.

## [NEW] FULL TUTORIAL 🚀 

### We now provide a full notebook tutorial on how to train/test/deploy your own SGG model! Please check it out:

[TUTORIAL.ipynb](docs/TUTORIAL.ipynb)

<!-- 
## Background

This implementation is a new benchmark for the task of Scene Graph Generation, based on a fork of the [SGG Benchmark by Kaihua Tang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). The implementation by Kaihua is a good starting point however it is very outdated and is missing a lot of new development for the task.
My goal with this new codebase is to provide an up-to-date and easy-to-run implementation of common approaches in the field of Scene Graph Generation. 
This codebase also focuses on real-time and real-world usage of Scene Graph Generation with dedicated dataset tools and a large choice of object detection backbones.
This codebase is actually a work-in-progress, do not expect everything to work properly on the first run. If you find any bugs, please feel free to post an issue or contribute with a PR. -->

## Recent Updates

- [X] 09/03/2026: **REACT++ released!** New YOLO12m-based model with improved accuracy and ONNX export for ~2× faster inference (13.4ms). See [MODEL_ZOO.md](docs/MODEL_ZOO.md) for weights and results.
- [X] 09/03/2026:  🤗 As an effort to push the open-source community in SGG we are releasing the PSG, VG150 and IndoorVG datasets on the huggingface hub! Please see [DATASET.md](docs/DATASET.md) for more details.
- [X] 09/03/2026: The codebase now support YOLO26, the new YOLO release from [ultralytics](https://github.com/ultralytics/ultralytics).
- [X] 15/08/2025: I have created a new tool to annotate your own SGG dataset with visual relationships, please check it out: [SGG-Annotate](https://github.com/Maelic/SGG-Annotate). More info in [ANNOTATIONS.md](docs/ANNOTATIONS.md).
- [X] 31.07.2025: REACT has been accepted at the BMVC 2025 conference!
- [X] 26.05.2025: I have added some explanation for two new metrics: InformativeRecall@K and Recall@K Relative. InformativeRecall@K is defined in [Mining Informativeness in Scene Graphs](https://www.sciencedirect.com/science/article/pii/S016786552500008X) and can help to measure the pertinence and robustness of models for real-world applications. Please check the [METRICS.md](docs/METRICS.md) file for more information.
- [X] 26.05.2025: The codebase now supports also YOLOV12, see [configs/hydra/VG/REACT++.yaml](configs/hydra/VG/REACT++.yaml).
- [X] 04.12.2024: Official release of the REACT model weights for VG150, please see [MODEL_ZOO.md](docs/MODEL_ZOO.md)
- [X] 03.12.2024: Official release of the [REACT model](https://arxiv.org/abs/2405.16116)
- [X] 23.05.2024: Added support for Hyperparameters Tuning with the RayTune library, please check it out: [Hyperparameters Tuning](#hyperparameters-tuning)
- [X] 23.05.2024: Added support for the YOLOV10 backbone and SQUAT relation head!
- [X] 28.05.2024: Official release of our [Real-Time Scene Graph Generation](https://arxiv.org/abs/2405.16116) implementation.
- [X] 23.05.2024: Added support for the [YOLO-World](https://www.yoloworld.cc/) backbone for Open-Vocabulary object detection!
- [X] 10.05.2024: Added support for the [PSG Dataset](https://github.com/Jingkang50/OpenPSG)
- [X] 03.04.2024: Added support for the IETrans method for data augmentation on the Visual Genome dataset, please check it out! [IETrans](./process_data/data_augmentation/README.md).
- [X] 03.04.2024: Update the demo, now working with any models, check [DEMO.md](./demo/).
- [X] 01.04.2024: Added support for Wandb for better visualization during training, tutorial coming soon.

## Contents

1. [Quick Start](#quick-start-)
2. [Full Tutorial Notebook](#full-tutorial-)
3. [Installation](docs/INSTALL.md)
4. [Datasets Preparation](docs/DATASET.md)
5. [Model Zoo & Weights](docs/MODEL_ZOO.md)
6. [Supported Models & Backbones](#supported-models)
7. [Metrics and Results](docs/METRICS.md)
8. [Training Instructions](#perform-training-on-scene-graph-generation)
9. [Hyperparameters Tuning](#hyperparameters-tuning)
10. [Evaluation Instructions](#evaluation)
11. [Citations](#citations)

<!-- ## Overview

Note from [Kaihua Tang](https://github.com/KaihuaTang), I keep it for reference:

" This project aims to build a new CODEBASE of Scene Graph Generation (SGG), and it is also a Pytorch implementation of the paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949). The previous widely adopted SGG codebase [neural-motifs](https://github.com/rowanz/neural-motifs) is detached from the recent development of Faster/Mask R-CNN. Therefore, I decided to build a scene graph benchmark on top of the well-known [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) project and define relationship prediction as an additional roi_head. By the way, thanks to their elegant framework, this codebase is much more novice-friendly and easier to read/modify for your own projects than previous neural-motifs framework (at least I hope so). It is a pity that when I was working on this project, the [detectron2](https://github.com/facebookresearch/detectron2) had not been released, but I think we can consider [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) as a more stable version with less bugs, hahahaha. I also introduce all the old and new metrics used in SGG, and clarify two common misunderstandings in SGG metrics in [METRICS.md](METRICS.md), which cause abnormal results in some papers. " -->

## Installation

Check [INSTALL.md](docs/INSTALL.md) for installation instructions.

## Datasets

Check [DATASET.md](docs/DATASET.md) for instructions regarding dataset preprocessing, including how to create your own dataset with [SGG-Annotate](https://github.com/Maelic/SGG-Annotate).

## Supported Models

### Background 

Scene Graph Generation approaches can be categorized between one-stage and two-stage approaches:
1. **Two-stages approaches** are the original implementation of SGG. It decouples the training process into (1) training an object detection backbone and (2) using bounding box proposals and image features from the backbone to train a relation prediction model.
2. **One-stage approaches** are learning both the object and relation features in the same learning stage. ***This codebase focuses only on the first category, two-stage approaches***.

### Object Detection Backbones

We proposed different object detection backbones that can be plugged with any relation prediction head, depending on the use case.

:rocket: NEW! No need to train a backbone anymore, we support Yolo-World for fast and easy open-vocabulary inference. Please check it out!

- [x] [YOLO26](https://docs.ultralytics.com/models/yolo26/): New yolo architecture for SOTA real-time object detection.
- [x] [YOLO12](https://docs.ultralytics.com/models/yolo12/): New yolo architecture for SOTA real-time object detection.
- [x] [YOLO11](https://docs.ultralytics.com/models/yolo11/): New  yolo version from Ultralytics for SOTA real-time object detection.
- [x] [YOLOV10](https://docs.ultralytics.com/models/yolov10/): New end-to-end yolo architecture for SOTA real-time object detection.
- [x] [YOLOV8-World](https://docs.ultralytics.com/models/yolo-world/): SOTA in real-time open-vocabulary object detection!
- [x] [YOLOV9](https://docs.ultralytics.com/models/yolov9/): SOTA in real-time object detection.
- [x] [YOLOV8](https://docs.ultralytics.com/models/yolov8/): New  yolo version from Ultralytics for SOTA real-time object detection.
- [x] **LEGACY** Faster-RCNN: This is the original backbone used in most SGG approaches. It is based on a ResNeXt-101 feature extractor and an RPN for regression and classification. See [the original paper for reference](https://arxiv.org/pdf/1506.01497.pdf). Performance is 38.52/26.35/28.14 mAp on VG train/val/test set respectively. You can find the original pretrained model by Kaihua [here](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw).

### Relation Heads

We try to compiled the main approaches for relation modeling in this codebase:

- [x] REACT++ (2025): [REACT++: Efficient Cross-Attention for Real-Time Scene Graph Generation](https://arxiv.org/abs/2603.06386). Improved version of REACT with a new low-cost relation head and YOLO12 as backbone. **Best results on PSG, IndoorVG and VG150.** ONNX export available for deployment at ~55 FPS. Weights at [MODEL_ZOO.md](docs/MODEL_ZOO.md).

- [x] REACT (2025): [REACT: Real-time Efficiency and Accuracy Compromise for Tradeoffs in Scene Graph Generation](https://arxiv.org/abs/2405.16116)

- [x] SQUAT (2023): [Devil's on the Edges: Selective Quad Attention for Scene Graph Generation](https://arxiv.org/abs/2304.03495), thanks to the [official implementation by authors](https://github.com/hesedjds/SQUAT)

- [x] PE-NET (2023): [Prototype-based Embedding Network for Scene Graph Generation](https://arxiv.org/abs/2303.07096), thanks to the [official implementation by authors](https://github.com/VL-Group/PENET)

- [x] SHA-GCL (2022): [Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation in Pytorch](https://arxiv.org/abs/2203.09811), thanks to the [official implementation by authors](https://github.com/dongxingning/SHA-GCL-for-SGG)

- [x] GPS-NET (2020): [GPS-Net: Graph Property Sensing Network for Scene Graph Generation](https://arxiv.org/abs/2003.12962), thanks to the [official implementation by authors](https://github.com/siml3/GPS-Net)

- [x] Transformer (2020): [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), thanks to the [implementation by Kaihua](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

- [x] VCTree (2018): [Learning to Compose Dynamic Tree Structures for Visual Contexts](https://arxiv.org/abs/1812.01880), thanks to the [implementation by Kaihua](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

- [x] Neural-Motifs (2018): [Neural Motifs: Scene Graph Parsing with Global Context](https://arxiv.org/abs/1711.06640), thanks to the [implementation by Kaihua](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

- [x] IMP (2017): [Scene Graph Generation by Iterative Message Passing](https://arxiv.org/abs/1701.02426), thanks to the [implementation by Kaihua](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

### Debiasing methods

On top of relation heads, several debiasing methods have been proposed through the years with the aim of increasing the accuracy of baseline models in the prediction of tail classes.

- [x] Hierarchical (2024): [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842), thanks to the [implementation by authors](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch)

- [x] Causal (2020): [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), thanks to the [implementation by authors](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

### Data Augmentation methods

Due to severe biases in datasets, the task of Scene Graph Generation as also been tackled through data-centring approaches.

- [x] IETrans (2022): [Fine-Grained Scene Graph Generation with Data Transfer](https://arxiv.org/abs/2203.11654), custom implementation based on the one [by Zijian Zhou](https://github.com/franciszzj/HiLo/tree/main/tools/data_prepare)

### Model ZOO

We provide some of the pre-trained weights for evaluation or usage in downstream tasks, please see [MODEL_ZOO.md](docs/MODEL_ZOO.md).

## Metrics and Results **(IMPORTANT)**
Explanation of metrics in our toolkit and reported results are given in [METRICS.md](docs/METRICS.md)

## REACT++ Quick Start

REACT++ is our best model for real-time SGG, combining the YOLO12m detector with an efficient relation head. Pretrained weights and ONNX models are available in [MODEL_ZOO.md](docs/MODEL_ZOO.md).

### Training REACT++ on PSG

```bash
python tools/relation_train_net_hydra.py --config-name PSG/REACT++ --task sgdet --save-best
```

### Evaluating REACT++ (PyTorch)

```bash
python tools/relation_eval_hydra.py --run-dir checkpoints/PSG/react++_yolo12m --task sgdet
```

### Exporting to ONNX

```bash
python tools/export_onnx.py --run-dir checkpoints/PSG/react++_yolo12m
```

### Evaluating the ONNX model on PSG

This runs the full SGDet evaluation on the PSG test set using ONNX Runtime (GPU by default):

```bash
python tools/eval_onnx_psg.py --run-dir checkpoints/PSG/react++_yolo12m --provider CUDAExecutionProvider
```

Results are saved to `checkpoints/PSG/react++_yolo12m/inference_onnx/onnx_eval_summary.json`.

## YOLOV8/9/10/11/12/World Pre-training

If you want to use YoloV8/9/10/11/12 or Yolo-World as a backbone instead of Faster-RCNN, you need to first train a model using the official [ultralytics implementation](https://github.com/ultralytics/ultralytics). To help you with that, I have created a [dedicated notebook](process_data/convert_to_yolo.ipynb) to generate annotations in YOLO format from a .h5 file (SGG format). 
Once you have a model, you can modify [a config file](configs/hydra) and change the path `pretrained_detector_ckpt` to your model weights. Please note that you will also need to change the variable `yolo.size` and `yolo.out_channels` accordingly if you use another variant of YOLO (nano, small or large for instance). 
For training an SGG model with YOLO as a backbone, you need to modify the `meta_architecture` variable in the same config file to `GeneralizedYOLO`. You can then follow the standard procedure for training below.

### Custom-domain models (new)

For a complete, verified end-to-end walkthrough of building an SGG model on **your own
domain** — data collection, geometric relation synthesis, YOLO-World backbone
fine-tuning, training, evaluation and ONNX deployment (worked example: construction-site
safety with `person / helmet / safety-vest` and a `wearing` relation, reaching
R@100 0.98) — see **[train_custom_model.md](train_custom_model.md)**.
A **closed-vocabulary YOLO12** variant of the same walkthrough (much lighter backbone,
no CLIP vocabulary handling, measured comparison against YOLO-World) is available in
**[train_custom_model_yolo12.md](train_custom_model_yolo12.md)**.

## Faster R-CNN pre-training (legacy)

We do not support Faster-RCNN pre-training anymore.

## Perform training on Scene Graph Generation

There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use the argument ```--task``` to select the protocols. 

For **Predicate Classification (PredCls)**, we need to set:
``` bash
--task predcls
```
For **Scene Graph Classification (SGCls)**: :warning: SGCls mode is currently LEGACY and NOT SUPPORTED anymore for any YOLO-based model, please find the reason why [in this issue](https://github.com/Maelic/SGG-Benchmark/issues/45).
``` bash
--task sgcls
```
For **Scene Graph Detection (SGDet)**:
``` bash
--task sgdet
```

### Predefined Models
We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```. To select our predefined models, you can use ```MODEL.ROI_RELATION_HEAD.PREDICTOR```.

For [REACT++](https://arxiv.org/abs/2603.06386) Model:
```bash
model.roi_relation_head.predictor REACTPlusPlusPredictor
```

For [REACT](https://arxiv.org/abs/2405.16116v2) Model:
```bash
model.roi_relation_head.predictor  REACTPredictor
```

For [PE-NET](https://arxiv.org/abs/2303.07096) Model:
```bash
model.roi_relation_head.predictor PrototypeEmbeddingNetwork
```

For [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) Model:
```bash
model.roi_relation_head.predictor  MotifPredictor
```
For [Iterative-Message-Passing(IMP)](https://arxiv.org/abs/1701.02426) Model (Note that SOLVER.BASE_LR should be changed to 0.001 in SGCls, or the model won't converge):
```bash
model.roi_relation_head.predictor  IMPPredictor
```
For [VCTree](https://arxiv.org/abs/1812.01880) Model:
```bash
model.roi_relation_head.predictor VCTreePredictor
```
For Transformer Model (Note that Transformer Model needs to change SOLVER.BASE_LR to 0.001, SOLVER.SCHEDULE.TYPE to WarmupMultiStepLR, SOLVER.MAX_ITER to 16000, SOLVER.IMS_PER_BATCH to 16, SOLVER.STEPS to (10000, 16000).), which is provided by [Jiaxin Shi](https://github.com/shijx12):
```bash
model.roi_relation_head.predictor TransformerPredictor
```

### Examples of the Training Command

> **Recommended approach**: Use the Hydra-based training script `tools/relation_train_net_hydra.py` with configs from `configs/hydra/`. See the [REACT++ Quick Start](#react-quick-start) section for an example.

## Hyperparameters Tuning

Required library:
```pip install ray[data,train,tune] optuna tensorboard```

We provide a training loop for hyperparameters tuning in [hyper_param_tuning.py](tools/hyper_param_tuning.py). This script uses the [RayTune](https://docs.ray.io/en/latest/tune/index.html) library for efficient hyperparameters search. You can define a ```search_space``` object with different values related to the optimizer (AdamW and SGD supported for now) or directly customize the model structure with model parameters (for instance Linear layers dimensions or MLP dimensions etc). The ```ASHAScheduler``` scheduler is used for the early stopping of bad trials. The default value to optimize is the overall loss but this can be customize to specific loss values or standard metrics such as ```mean_recall```.

To launch the script, do as follow:

```
CUDA_VISIBLE_DEVICES=0 python tools/hyper_param_tuning.py --save-best --task sgdet --config-file "./configs/hydra/IndoorVG/REACT++.yaml"
```

The config and OUTPUT_DIR paths need to be absolute to allow faster loading. A lot of terminal outputs are disabled by default during tuning, using the ```cfg.VERBOSE``` variable.

To watch the results with tensorboardX: 
```
tensorboard --logdir=./ray_results/train_relation_net_2024-06-23_15-28-01
```

## Evaluation

### Recommended Approach (Hydra-based)

For REACT++ and any model trained with the Hydra pipeline, evaluation is done with `tools/relation_eval_hydra.py` by pointing it at a checkpoint directory:

```bash
# SGDet evaluation (PSG)
python tools/relation_eval_hydra.py --run-dir checkpoints/PSG/react++_yolo12m --task sgdet

# SGDet evaluation with a specific checkpoint
python tools/relation_eval_hydra.py --run-dir checkpoints/PSG/react++_yolo12m --task sgdet --checkpoint best_model_epoch_9.pth
```

See the [REACT++ Quick Start](#react-quick-start) section for full training/eval/ONNX export commands.

## Citations

If you find this project helps your research, please kindly consider citing our project or papers in your publications.


```
@misc{neau2026reactplusplus,
      title={REACT++: Efficient Cross-Attention for Real-Time Scene Graph Generation
}, 
      author={Maëlic Neau and Zoe Falomir},
      year={2026},
      eprint={2603.06386},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.06386}, 
}
```

```
@misc{neau2024reactrealtimeefficiencyaccuracy,
      title={REACT: Real-time Efficiency and Accuracy Compromise for Tradeoffs in Scene Graph Generation}, 
      author={Maëlic Neau and Paulo E. Santos and Anne-Gwenn Bosser and Cédric Buche},
      year={2024},
      eprint={2405.16116},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.16116}, 
}
```
