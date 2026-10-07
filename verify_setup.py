"""
EYEON - Phase 0 Setup & Environment Verification Script.

Run this script to verify that your environment, folder structure,
dependencies, model weights, and configuration are 100% ready.
"""

import sys
import os
from pathlib import Path

def print_status(component: str, ok: bool, message: str = ""):
    symbol = "[OK]" if ok else "[FAIL]"
    print(f" {symbol} {component:<25} {message}")

def verify_all():
    print("=" * 65)
    print("   EYEON Intelligent Tracker - Phase 0 Setup Verification")
    print("=" * 65)
    all_ok = True

    # 1. Python version check
    py_ver = sys.version_info
    py_ok = (py_ver.major == 3 and py_ver.minor >= 10)
    print_status(
        "Python Version",
        py_ok,
        f"{py_ver.major}.{py_ver.minor}.{py_ver.micro} (Requires >= 3.10)"
    )
    if not py_ok:
        all_ok = False

    # 2. Dependency imports
    packages = [
        ("ultralytics", "ultralytics"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("shapely", "shapely"),
        ("pyyaml", "yaml"),
        ("streamlit", "streamlit"),
        ("scipy", "scipy"),
        ("lapx", "lap"),
        ("sqlite3", "sqlite3"),
    ]

    for pkg_name, mod_name in packages:
        try:
            __import__(mod_name)
            print_status(f"Dependency: {pkg_name}", True, f"imported as '{mod_name}'")
        except ImportError as e:
            print_status(f"Dependency: {pkg_name}", False, str(e))
            all_ok = False

    # 3. Directory structure check
    required_dirs = [
        "src",
        "src/detection",
        "src/tracking",
        "src/association",
        "src/positioning",
        "src/risk",
        "src/alerts",
        "src/database",
        "src/utils",
        "dashboard",
        "data",
        "data/videos",
        "data/images",
        "data/annotations",
        "data/processed",
        "data/alerts",
        "models",
        "evaluation",
        "notebooks",
    ]

    root = Path(__file__).resolve().parent
    missing_dirs = [d for d in required_dirs if not (root / d).is_dir()]
    if not missing_dirs:
        print_status("Directory Structure", True, "All 19 subdirectories exist")
    else:
        print_status("Directory Structure", False, f"Missing: {missing_dirs}")
        all_ok = False

    # 4. Package inits
    init_packages = [
        "src",
        "src/detection",
        "src/tracking",
        "src/association",
        "src/positioning",
        "src/risk",
        "src/alerts",
        "src/database",
        "src/utils",
        "dashboard",
        "evaluation",
    ]
    missing_inits = [p for p in init_packages if not (root / p / "__init__.py").is_file()]
    if not missing_inits:
        print_status("Python Packages", True, "All __init__.py files present")
    else:
        print_status("Python Packages", False, f"Missing __init__.py in: {missing_inits}")
        all_ok = False

    # 5. Config Loader check
    try:
        from src.utils.config_loader import load_config, get_path
        cfg = load_config()
        proj_name = cfg.get("project", {}).get("name", "Unknown")
        print_status("Configuration", True, f"config.yaml valid (Project: '{proj_name}')")
    except Exception as e:
        print_status("Configuration", False, f"Config load failed: {e}")
        all_ok = False

    # 6. Model Weight check
    model_paths = [root / "models" / "yolo11n.pt", root / "yolo11n.pt"]
    found_model = any(p.is_file() for p in model_paths)
    if found_model:
        size_mb = max(p.stat().st_size for p in model_paths if p.is_file()) / (1024 * 1024)
        print_status("Model Weights", True, f"yolo11n.pt found ({size_mb:.2f} MB)")
    else:
        print_status("Model Weights", False, "yolo11n.pt not found in models/ or root")
        all_ok = False

    # 7. Test Videos check
    videos_dir = root / "data" / "videos"
    sample_videos = list(videos_dir.glob("*.mp4")) if videos_dir.exists() else []
    if sample_videos:
        print_status("Test Videos", True, f"{len(sample_videos)} test videos available in data/videos")
    else:
        print_status("Test Videos", False, "No test videos found in data/videos")
        all_ok = False

    # 8. Git ignore check
    gitignore_file = root / ".gitignore"
    print_status(".gitignore", gitignore_file.is_file(), "Present in project root" if gitignore_file.is_file() else "Missing")

    print("=" * 65)
    if all_ok:
        print("  🎉 PHASE 0 SETUP IS 100% COMPLETE AND VERIFIED!")
    else:
        print("  ⚠️  SOME ITEMS REQUIRE ATTENTION BEFORE MOVING TO PHASE 1.")
    print("=" * 65)
    return all_ok

if __name__ == "__main__":
    success = verify_all()
    sys.exit(0 if success else 1)
