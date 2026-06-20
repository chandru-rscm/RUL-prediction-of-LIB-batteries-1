import sys

print(f"Python version: {sys.version}")
print("-" * 40)

packages = [
    ("numpy", "np"),
    ("pandas", "pd"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("h5py", "h5py"),
    ("tqdm", "tqdm"),
    ("shap", "shap"),
]

all_good = True
for pkg, label in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "installed")
        print(f"  ✓  {label:<20} {version}")
    except ImportError:
        print(f"  ✗  {label:<20} NOT FOUND — run: pip install {label}")
        all_good = False

print("-" * 40)
if all_good:
    print("All good! Ready to start.")
else:
    print("Some packages missing. Install them first.")