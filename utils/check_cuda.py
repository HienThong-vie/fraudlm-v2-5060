"""Quick RTX/PyTorch compatibility and memory check."""

import sys

import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build, not the CPU-only wheel.")
props = torch.cuda.get_device_properties(0)
gb = props.total_memory / 1024**3
print(f"GPU: {props.name}")
print(f"VRAM: {gb:.2f} GB")
print(f"Compute capability: {props.major}.{props.minor}")
if gb < 7.5:
    print("WARNING: less than 7.5 GB is visible; close GPU applications before training.")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print(f"FP16 CUDA test: {(x @ x).mean().item():.4f} (OK)")
