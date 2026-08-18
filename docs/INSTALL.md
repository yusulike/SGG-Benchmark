## Installation (uv-based)

This project supports a fast, reproducible install using `uv` and a local `.venv`.

Prerequisites
- Python 3.8+ (3.11 recommended)
- System NVIDIA drivers and CUDA runtime if you plan to use GPU acceleration

Overview
- Use `./scripts/install_uv.sh` to create a local `.venv`, install pure-Python dependencies, and provide guidance for GPU packages.
- We purposely do not pin platform-dependent packages like `torch` or `tensorrt` in the project lockfile — see the Lockfile section.

Quick install (one-liner)

```bash
chmod +x scripts/install_uv.sh
./scripts/install_uv.sh
```

What the installer does
- Installs `uv` if missing and creates `.venv` with Python 3.11.
- Installs `torch>=2.0` and `torchvision>=0.15` (letting `uv` pick the best wheel for your platform).
- Installs the pure-Python dependencies in `requirements.txt`.
- Installs Ultralytics CLIP, ONNX tooling, and attempts `onnxruntime-gpu` (falls back to CPU `onnxruntime`).
- Attempts to install `tensorrt` via `pip` and prints manual install instructions if that fails.

Note on Ultralytics
- Use the **official** [ultralytics](https://github.com/ultralytics/ultralytics) package (`>=8.3.100`, see `requirements.txt`).
- Community YOLO12 forks (e.g. [`sunsmarterjie/yolov12`](https://github.com/sunsmarterjie/yolov12), pinned at 8.3.63) are **not compatible**: they lack YOLOE backbones, use different model YAML names (`yolov12.yaml` instead of `yolo12m.yaml`), and miss NMS APIs the codebase relies on.

Activating the environment

```bash
source .venv/bin/activate
# or run commands without activating with uv run:
uv run python tools/relation_train_net_hydra.py ...
```

PyTorch / CUDA / TensorRT guidance
- We require `torch>=2.0` for optimized SDPA / `torch.scaled_dot_product_attention` and `torch.compile` compatibility.
- Wheels for `torch` are CUDA- and platform-specific. If you need a particular CUDA build (e.g. cu121), install it explicitly after running the installer:

```bash
# Example for CUDA 12.1 (replace with the wheel appropriate for your GPU)
uv run pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.0" "torchvision>=0.15"
```

- TensorRT Python bindings are often provided by the NVIDIA package repositories or prebuilt wheels tied to a specific platform. The installer will attempt `uv run pip install tensorrt` and otherwise shows a link to NVIDIA's install guide.

ONNX and ONNX Runtime
- The repo contains `tools/export_onnx.py` and `demo/onnx_model.py`. The installer installs `onnx`, `onnx-simplifier`, and attempts `onnxruntime-gpu` with CPU fallback.
- If you need maximum performance with TensorRT, follow NVIDIA's TensorRT instructions and then run `scripts/setup_cuda_libs.sh` to expose vendor libraries.

## Reproducing the full environment using the provided `scripts/uv.lock` (exact pinned versions)

To reproduce that environment exactly, follow the steps below. Note this will only succeed unchanged on machines with the same OS/architecture and matching CUDA runtime/drivers as the machine that produced the lockfile (CUDA 12.8, Driver 550.163.01).

```bash
# 1) Create a clean venv (do not run the full installer if you want exact pins)
uv venv --python 3.11
source .venv/bin/activate

# 2) Install the exact pinned packages from the lock (this will install the exact torch wheel recorded)
uv run pip install -r scripts/uv.lock

# 3) Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__)"

# 4) Install codebase
pip install .
```
