# deploy_diagnose.py
import os, sys, subprocess, json, glob
from pathlib import Path

print("=== PATH ===")
print("cwd:", os.getcwd())
print()

# 1) list large files (top 30)
print("=== TOP FILES (largest 30) ===")
files = []
for root, dirs, filenames in os.walk("."):
    for f in filenames:
        p = Path(root) / f
        try:
            size = p.stat().st_size
        except:
            size = 0
        files.append((size, str(p)))
files = sorted(files, reverse=True)[:30]
for size, p in files:
    mb = size / (1024**2)
    print(f"{mb:8.2f} MB\t{p}")
print()

# 2) show requirements snippet (first 200 lines if exists)
req = Path("requirements.txt")
print("=== requirements.txt (first 200 lines) ===")
if req.exists():
    with req.open() as f:
        for i, line in enumerate(f):
            if i>=200: break
            print(line.rstrip())
else:
    print("requirements.txt not found")
print()

# 3) check python packages: torch, ultralytics, onnxruntime
print("=== IMPORT CHECKS ===")
for pkg in ("torch", "ultralytics", "onnxruntime", "onnx", "opencv"):
    try:
        m = __import__(pkg)
        print(f"{pkg}: INSTALLED, version: {getattr(m,'__version__', 'unknown')}")
    except Exception as e:
        print(f"{pkg}: NOT INSTALLED ({e.__class__.__name__})")
print()

# 4) If torch is installed and CUDA available, show device info and try a tiny load if weights present
try:
    import torch
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        # report current memory allocated
        print("torch.cuda.memory_allocated(0):", torch.cuda.memory_allocated(0))
except Exception as e:
    print("torch check error:", e)
print()

# 5) Try to locate common model weight files (.pt, .pth, .onnx)
print("=== MODEL WEIGHTS FOUND ===")
weights = []
for ext in ("*.pt","*.pth","*.onnx","*.ckpt"):
    weights += glob.glob(f"**/{ext}", recursive=True)
weights = sorted(weights, key=lambda p: -Path(p).stat().st_size)[:50]
if weights:
    for p in weights:
        s = Path(p).stat().st_size / (1024**2)
        print(f"{s:8.2f} MB\t{p}")
else:
    print("No common weight files found (.pt/.pth/.onnx/.ckpt) in repo")
print()

# 6) (Optional) If ultralytics is importable, try to instantiate YOLO with a local weight (non-blocking)
if "ultralytics" in sys.modules or Path("requirements.txt").exists():
    try:
        import importlib
        spec = importlib.util.find_spec("ultralytics")
        if spec:
            print("ultralytics module available.")
            # attempt to find any .pt and print a warning if large
            if weights:
                print("Found weight(s) above; NOT attempting to load model automatically (could be heavy).")
            else:
                print("No weights found; skipping model load.")
        else:
            print("ultralytics not installed.")
    except Exception as e:
        print("ultralytics check error:", e)
print()

# 7) Print Python version + dockerfile existence
print("=== ENV / DOCKER ===")
print("Python:", sys.version.replace('\\n',' '))
print("Dockerfile exists:", Path("Dockerfile").exists())
print()

# 8) Quick pip freeze (first 200 lines) if pip available
print("=== PIP LIST (first 200 lines) ===")
try:
    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.STDOUT, text=True)
    for i, line in enumerate(out.splitlines()):
        if i>=200: break
        print(line)
except Exception as e:
    print("pip freeze failed:", e)
