#!/usr/bin/env python
"""Interactive runner for the SAFETY custom-domain SGG pipeline (yolo12).

Wraps every stage of train_custom_model.md in one place, detects what is
already done (smart resume), and prompts for the few knobs that matter:

  [1] setup     - create .venv-sgg via uv (torch cu128 + requirements + CLIP + onnx)
  [2] download  - HF VincentGOURBIN/ppe-detection (retry loop, xet disabled)
  [3] convert   - build_safety_coco.py -> datasets/SAFETY COCO-SG + _data/safety-yolo
  [4] backbone  - prepare_safety_yolo12_backbone.py (yolo12n/s/m/l)
  [5] train     - relation_train_net_hydra.py --config-name SAFETY/REACT_yolo12
  [6] eval      - relation_eval_hydra.py --run-dir ... --task sgdet
  [7] export    - export_onnx.py + test_react_world_onnx.py smoke test

Usage:
  python scripts/safety_pipeline.py                # interactive menu
  python scripts/safety_pipeline.py --stage all    # hands-free, skips done stages
  python scripts/safety_pipeline.py --stage status # just print stage status
  python scripts/safety_pipeline.py --stage backbone --size yolo12m --epochs 30 --yes

Run it with the repo venv interpreter once setup is done:
  .venv-sgg/Scripts/python.exe scripts/safety_pipeline.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)

VENV = REPO / ".venv-sgg"
VPY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

HF_REPO = "VincentGOURBIN/ppe-detection"
DATA_RAW = REPO / "_data" / "ppe-detection"
DATA_YOLO = REPO / "_data" / "safety-yolo"
COCO_DIR = REPO / "datasets" / "SAFETY" / "coco_format"
BACKBONE_DIR = REPO / "checkpoints" / "BACKBONES" / "SAFETY"
RUN_DIR = REPO / "checkpoints" / "SAFETY" / "react_yolo12"
CONFIG_NAME = "SAFETY/REACT_yolo12"
CLASS_MAP = ["3:person", "0:helmet", "4:vest"]  # source ids -> 3 target classes

TORCH_INDEX_DEFAULT = "https://download.pytorch.org/whl/cu128"
CLIP_GIT = "https://github.com/openai/CLIP.git"

DOWNLOAD_SNIPPET = """
import time
from huggingface_hub import snapshot_download
for attempt in range(10):
    try:
        p = snapshot_download(repo_id={repo!r}, repo_type='dataset',
                              local_dir={dst!r}, max_workers=4)
        print('downloaded to', p)
        break
    except Exception as e:
        print('retry:', e, flush=True)
        time.sleep(30)
else:
    raise SystemExit('download failed after 10 attempts')
"""

# ── small io helpers ────────────────────────────────────────────────────────

def ask(prompt, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        die("stdin is closed - run a stage non-interactively instead, "
            "e.g. --stage status or --stage all --yes")
    return val or (str(default) if default is not None else "")


def confirm(prompt, default=True):
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        val = input(f"{prompt}{suffix}: ").strip().lower()
    except EOFError:
        die("stdin is closed - run a stage non-interactively instead, "
            "e.g. --stage status or --stage all --yes")
    if not val:
        return default
    return val in ("y", "yes")


def die(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def sh(cmd, env_extra=None):
    """Run a stage command in the repo root with live output."""
    env = {**os.environ, **(env_extra or {})}
    proc = subprocess.run([str(c) for c in cmd], cwd=str(REPO), env=env)
    if proc.returncode != 0:
        die(f"command failed (exit {proc.returncode}): "
            f"{' '.join(str(c) for c in cmd)}")
    return proc.returncode


def maybe(prompt, default):
    """Prompt unless --yes; empty answer means 'use default'."""
    if ARGS.yes:
        return default
    return ask(prompt, default)


# ── stage status detection ──────────────────────────────────────────────────

def interpreter_ready(python=PY):
    probe = "import torch, ultralytics, clip, hydra"
    return subprocess.run(
        [str(python), "-c", probe], capture_output=True).returncode == 0


def stage_status():
    def exists(p):
        return Path(p).exists()

    checks = [
        ("setup    venv + torch/ultralytics/clip",
         VENV.exists() and interpreter_ready(VPY)),
        ("download raw dataset (data.yaml + train images)",
         exists(DATA_RAW / "data.yaml")
         and any((DATA_RAW / "train" / "images").glob("*"))),
        ("convert  COCO-SG jsons",
         all(exists(COCO_DIR / s / "_annotations.coco.json")
             for s in ("train", "val", "test"))),
        (f"backbone {ARGS.size}_backbone.pt",
         exists(BACKBONE_DIR / f"{ARGS.size}_backbone.pt")),
        ("train    model_epoch_*.pth",
         any(RUN_DIR.glob("model_epoch_*.pth"))),
        ("eval     inference_sgdet/ results",
         any((RUN_DIR / "inference_sgdet").glob("*")) if exists(RUN_DIR / "inference_sgdet") else False),
        ("export   react_yolo12.onnx",
         any(RUN_DIR.glob("*.onnx"))),
    ]
    return checks


def print_status():
    print(f"\nSAFETY pipeline status (repo: {REPO})")
    print("-" * 60)
    for label, done in stage_status():
        print(f"  [{'x' if done else ' '}] {label}")
    print()


# ── stages ──────────────────────────────────────────────────────────────────

def stage_setup():
    if VENV.exists() and interpreter_ready(VPY):
        if not confirm("venv already ready - reinstall packages?", False):
            return
    if not shutil_which("uv"):
        die("uv not found - install it from https://docs.astral.sh/uv/ first")
    index = maybe(f"torch wheel index", TORCH_INDEX_DEFAULT)
    sh(["uv", "venv", str(VENV), "--python", "3.12"])
    sh(["uv", "pip", "install", "--python", VPY,
        "torch", "torchvision", "--index-url", index])
    sh(["uv", "pip", "install", "--python", VPY, "-r", "requirements.txt"])
    sh(["uv", "pip", "install", "--python", VPY, "-e", ".", "--no-deps"])
    sh(["uv", "pip", "install", "--python", VPY, f"git+{CLIP_GIT}"])
    sh(["uv", "pip", "install", "--python", VPY,
        "onnx", "onnxscript", "onnxruntime-gpu"])
    print(f"\nsetup complete - rerun this script with:\n  {VPY} {Path(__file__).name}")


def shutil_which(name):
    from shutil import which
    return which(name)


def stage_download():
    if (DATA_RAW / "data.yaml").exists() and not ARGS.force:
        if not confirm("raw dataset exists - re-download?", False):
            return
    code = DOWNLOAD_SNIPPET.format(repo=HF_REPO, dst=str(DATA_RAW))
    sh([PY, "-c", code], env_extra={"HF_HUB_DISABLE_XET": "1"})


def stage_convert():
    cmd = [PY, "process_data/build_safety_coco.py",
           "--src", DATA_RAW, "--class-map", *CLASS_MAP,
           "--emit-yolo", DATA_YOLO]
    sh(cmd)
    # train_custom_model.md pitfall #2: stale statistics cache after data changes
    for cache in RUN_DIR.glob("*.cache"):
        cache.unlink()
        print(f"deleted stale cache: {cache}")


def stage_backbone():
    size = maybe("backbone size (yolo12n/s/m/l)", ARGS.size)
    epochs = int(maybe("fine-tune epochs", ARGS.bb_epochs))
    batch = int(maybe("batch", ARGS.bb_batch))
    data_yaml = DATA_YOLO / "data.yaml"
    if not data_yaml.exists():
        die(f"{data_yaml} missing - run the convert stage first")
    cmd = [PY, "scripts/prepare_safety_yolo12_backbone.py",
           "--size", size, "--data", data_yaml,
           "--epochs", epochs, "--batch", batch]
    print(f"\nbackbone fine-tune: {size}, {epochs} epochs, batch {batch}")
    sh(cmd)
    print(f"\nbackbone saved: {BACKBONE_DIR / (size + '_backbone.pt')}")


def stage_train():
    if not (BACKBONE_DIR / f"{ARGS.size}_backbone.pt").exists():
        die(f"backbone missing - run the backbone stage first")
    epochs = maybe("solver.max_epoch (empty = config default 20)", ARGS.train_epochs)
    ims = maybe("solver.ims_per_batch (empty = config default 8)", ARGS.ims)
    cmd = [PY, "tools/relation_train_net_hydra.py",
           "--config-name", CONFIG_NAME, "--task", ARGS.task]
    if epochs:
        cmd.append(f"solver.max_epoch={epochs}")
    if ims:
        cmd.append(f"solver.ims_per_batch={ims}")
    print(f"\nSGG training: {CONFIG_NAME} (task={ARGS.task})")
    sh(cmd)
    print(f"\ncheckpoints: {RUN_DIR}")


def stage_eval():
    if not any(RUN_DIR.glob("model_epoch_*.pth")):
        die(f"no model_epoch_*.pth in {RUN_DIR} - run the train stage first")
    cmd = [PY, "tools/relation_eval_hydra.py",
           "--run-dir", RUN_DIR, "--task", ARGS.task]
    sh(cmd)


def stage_export():
    ckpts = sorted(RUN_DIR.glob("model_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        die(f"no model_epoch_*.pth in {RUN_DIR} - run the train stage first")
    images = sorted((COCO_DIR / "test").glob("*.jpg"))
    if not images:
        die(f"no test images under {COCO_DIR / 'test'}")
    image = maybe("smoke-test image", str(images[0]))
    onnx = RUN_DIR / "react_yolo12.onnx"
    sh([PY, "tools/export_onnx.py", "--run-dir", RUN_DIR,
        "--image", image, "--onnx-path", onnx])
    print(f"\nONNX smoke test: {onnx}")
    sh([PY, "tools/test_react_world_onnx.py", "--onnx", onnx, "--source", image])
    print(f"""
exported: {onnx}
Output contract (train_custom_model.md section 8):
  boxes [N,6] = (x1,y1,x2,y2,label,score)  label: 1=person 2=helmet 3=safety-vest
  rels  [M,5] = (subj_idx,obj_idx,label,triplet_score,rel_score)  label: 1=wearing
Filter relations by relative rank / triplet_score, not an absolute threshold
(absolute rel_score stays ~1e-3). Violation rule: a person with no outgoing
`wearing` edge to a helmet/vest (section 9).""")


STAGES = {
    "setup": (stage_setup, "create .venv-sgg (uv + torch cu + requirements + CLIP + onnx)"),
    "download": (stage_download, "HF ppe-detection snapshot (retry loop)"),
    "convert": (stage_convert, "build_safety_coco.py -> COCO-SG + cleaned YOLO"),
    "backbone": (stage_backbone, "yolo12 backbone fine-tune"),
    "train": (stage_train, "REACT_yolo12 SGG training"),
    "eval": (stage_eval, "sgdet evaluation on the test split"),
    "export": (stage_export, "ONNX export + smoke test"),
}
ORDER = ["setup", "download", "convert", "backbone", "train", "eval", "export"]


def run_all():
    keys = dict(zip(ORDER, [ok for _, ok in stage_status()]))
    for name in ORDER:
        if keys[name] and not ARGS.force:
            print(f"== {name}: already done, skipping (use --force to redo)")
            continue
        if not ARGS.yes and not confirm(f"\nrun stage '{name}'?", True):
            continue
        STAGES[name][0]()


def menu():
    while True:
        print_status()
        print("  [a] run whole pipeline (smart resume)")
        for i, name in enumerate(ORDER, 1):
            print(f"  [{i}] {name:9s} - {STAGES[name][1]}")
        print("  [s] status only")
        print("  [q] quit")
        choice = ask("select").lower()
        if choice in ("q", ""):
            return
        if choice == "a":
            run_all()
        elif choice == "s":
            continue
        elif choice.isdigit() and 1 <= int(choice) <= len(ORDER):
            STAGES[ORDER[int(choice) - 1]][0]()
        input("\npress Enter to continue...")


def ensure_venv_hop():
    """If the current interpreter lacks deps but .venv-sgg has them, delegate there.

    Uses subprocess instead of os.execv: on Windows, exec-replacing a process
    spawned by uv/cmd can lose the console stdin handle, which kills input()
    prompts (the menu dies and keystrokes leak into the parent shell).
    """
    if interpreter_ready():
        return
    if VPY.exists() and interpreter_ready(VPY):
        print(f"switching to repo venv: {VPY}")
        sys.exit(subprocess.run(
            [str(VPY), str(Path(__file__).resolve()), *sys.argv[1:]]).returncode)


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="menu",
                    choices=["menu", "all", "status", *ORDER],
                    help="run a single stage, 'all', 'status', or the menu")
    ap.add_argument("--yes", action="store_true",
                    help="non-interactive: accept all defaults")
    ap.add_argument("--force", action="store_true",
                    help="rerun stages even if their outputs exist")
    ap.add_argument("--size", default="yolo12m",
                    help="backbone size (yolo12n/s/m/l)")
    ap.add_argument("--bb-epochs", default="30", help="backbone fine-tune epochs")
    ap.add_argument("--bb-batch", default="16", help="backbone fine-tune batch")
    ap.add_argument("--train-epochs", default="",
                    help="SGG solver.max_epoch override (empty = config default)")
    ap.add_argument("--ims", default="",
                    help="SGG solver.ims_per_batch override (empty = config default)")
    ap.add_argument("--task", default="sgdet", help="eval task (sgdet/predcls/sgcls)")
    return ap.parse_args()


ARGS = None  # set in main(); module-level so maybe() can see it


def main():
    global ARGS
    ARGS = parse_args()
    ensure_venv_hop()
    if ARGS.stage == "status":
        print_status()
    elif ARGS.stage == "menu":
        menu()
    elif ARGS.stage == "all":
        run_all()
    else:
        STAGES[ARGS.stage][0]()


if __name__ == "__main__":
    main()
