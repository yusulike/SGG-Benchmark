## Installation (uv-based)

This project supports a fast, reproducible install using `uv` and a local `.venv`.

Prerequisites
- Python 3.11+ (3.12 tested)
- System NVIDIA drivers and CUDA runtime if you plan to use GPU acceleration

Quick install

The root `pyproject.toml` defines the **full** environment — including GPU
`torch`/`torchvision` from the CUDA 12.1 wheel index and the Ultralytics CLIP
fork — so a single sync is enough:

```bash
# install uv if missing: https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

Then run everything through `uv run`:

```bash
uv run python tools/relation_train_net_hydra.py ...
```

or activate the environment first (`source .venv/bin/activate`, or
`.venv\Scripts\activate` on Windows) and call `python` directly.

What `uv sync` does
- Creates `.venv` (Python >= 3.11) if missing and installs every dependency from the root `uv.lock`, `torch 2.5.1+cu121` included.
- Installs the `sgg_benchmark` package itself in editable mode — required because some `tools/` scripts import it before their own `sys.path` bootstrap.
- Also installs the serving/streaming dependencies (`fastapi`, `uvicorn`, `gstreamer-bundle`); they are harmless if you do not use them.
- TensorRT is **not** included (platform-specific wheels) — see below.
- `torchtext` is **not** included: it only serves the optional `image_retrieval` feature and has no Python >= 3.12 wheels. On Python 3.11, `uv pip install torchtext==0.18.0` if you need it.

Note on Ultralytics
- Use the **official** [ultralytics](https://github.com/ultralytics/ultralytics) package (`>=8.3.100`, see `requirements.txt`).
- Community YOLO12 forks (e.g. [`sunsmarterjie/yolov12`](https://github.com/sunsmarterjie/yolov12), pinned at 8.3.63) are **not compatible**: they lack YOLOE backbones, use different model YAML names (`yolov12.yaml` instead of `yolo12m.yaml`), and miss NMS APIs the codebase relies on.

PyTorch / CUDA / TensorRT guidance
- We require `torch>=2.0` for optimized SDPA / `torch.scaled_dot_product_attention` and `torch.compile` compatibility.
- The lockfile pins `torch` from the `pytorch-cu121` index via `[[tool.uv.index]]` in `pyproject.toml`. To switch CUDA builds, change that index (e.g. `cu124`) and re-lock:

```bash
uv lock && uv sync
```

- TensorRT Python bindings are often provided by the NVIDIA package repositories or prebuilt wheels tied to a specific platform: follow NVIDIA's install guide, then run `scripts/setup_cuda_libs.sh` to expose vendor libraries.

ONNX and ONNX Runtime
- The environment includes `onnx` and `onnxruntime-gpu` (CPU `onnxruntime` can be swapped in on machines without GPU support).
- For maximum performance with TensorRT, follow NVIDIA's TensorRT instructions and then run `scripts/setup_cuda_libs.sh` to expose vendor libraries.

## Legacy alternatives

- `./scripts/install_uv.sh` still works: it creates `.venv` and `uv pip install`s `requirements.txt` piece by piece instead of using the project lockfile. Prefer `uv sync`.
- To reproduce the *upstream author's* exact environment, see `scripts/uv.lock` below.

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
