#!/usr/bin/env python3
"""Verify that the development environment matches required dependency versions.

Run on each machine to check for version mismatches:
    python verify_environment.py

Or compare two machines:
    python verify_environment.py --json > mac_env.json
    ssh space "cd .../RL-CollectiveTransport && python verify_environment.py --json" > ubuntu_env.json
    diff mac_env.json ubuntu_env.json
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys


# ── Required versions ──
# Update these when upgrading dependencies
REQUIRED = {
    "argos3": "4bb398cd",  # ilpincy/argos3 commit hash (post-beta59, includes simulation loop fix)
    "argos3_min_version": "3.0.0-beta59",
    "buzz": "0.0.1",
    "python_min": "3.10",
    "pytorch_min": "2.0",
    "zmq_min": "4.3",
}


def get_version(cmd, regex=None):
    """Run a command and extract a version string."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr
        if regex:
            match = re.search(regex, output)
            return match.group(1) if match else output.strip()
        return output.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_environment():
    """Check all dependencies and return a report."""
    report = {
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.machine()}",
        "checks": [],
    }

    def check(name, actual, expected=None, required=True):
        status = "ok" if actual else ("MISSING" if required else "optional-missing")
        if expected and actual and expected not in str(actual):
            status = "MISMATCH"
        report["checks"].append({
            "name": name,
            "actual": actual,
            "expected": expected,
            "status": status,
        })
        return status == "ok"

    # ARGoS
    argos_version = get_version(["argos3", "--version"], r"ARGOS_VERSION=(.+)")
    check("argos3", argos_version, REQUIRED["argos3_min_version"])

    # Check ARGoS git commit (if built from source with version info)
    argos_commit = get_version(
        ["argos3", "--version"],
        r"ARGOS_GIT_COMMIT=(.+)"
    )
    if argos_commit:
        check("argos3_commit", argos_commit, REQUIRED["argos3"])
    else:
        # Can't verify commit from installed binary — check build path
        report["checks"].append({
            "name": "argos3_commit",
            "actual": "unknown (binary doesn't embed commit hash)",
            "expected": REQUIRED["argos3"],
            "status": "WARN",
        })

    # Buzz
    buzz_version = get_version(["bzzc", "--version"], r"version (.+)")
    check("buzz", buzz_version, REQUIRED["buzz"])

    # Python
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    check("python", py_version, REQUIRED["python_min"])

    # PyTorch
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        device = "cuda" if cuda_available else ("mps" if mps_available else "cpu")
        check("pytorch", f"{torch_version} ({device})", REQUIRED["pytorch_min"])
    except ImportError:
        check("pytorch", None, REQUIRED["pytorch_min"])

    # ZMQ
    try:
        import zmq
        zmq_version = zmq.zmq_version()
        pyzmq_version = zmq.__version__
        check("zmq", f"libzmq={zmq_version} pyzmq={pyzmq_version}", REQUIRED["zmq_min"])
    except ImportError:
        check("zmq", None, REQUIRED["zmq_min"])

    # GSP-RL
    try:
        import gsp_rl
        gsp_version = getattr(gsp_rl, "__version__", "installed (no version)")
        check("gsp_rl", gsp_version)
    except ImportError:
        check("gsp_rl", None)

    # argos3-nonuniform-objects plugin
    plugin_path = shutil.which("argos3")
    if plugin_path:
        plugin_dir = os.path.join(os.path.dirname(plugin_path), "..", "lib", "argos3")
        nonuniform = os.path.exists(
            os.path.join(plugin_dir, "libargos3plugin_simulator_nonuniform_objects.so")
        ) or os.path.exists(
            os.path.join(plugin_dir, "libargos3plugin_simulator_nonuniform_objects.dylib")
        )
        check("argos3-nonuniform-objects", "installed" if nonuniform else None)
    else:
        check("argos3-nonuniform-objects", None)

    # CMake
    cmake_version = get_version(["cmake", "--version"], r"cmake version (.+)")
    check("cmake", cmake_version, "3.16", required=False)

    return report


def print_report(report):
    """Print a human-readable report."""
    print(f"Environment: {report['hostname']} ({report['platform']})")
    print(f"{'='*60}")
    all_ok = True
    for c in report["checks"]:
        symbol = {"ok": "✓", "MISSING": "✗", "MISMATCH": "⚠", "WARN": "?",
                  "optional-missing": "-"}.get(c["status"], "?")
        line = f"  {symbol} {c['name']}: {c['actual'] or 'NOT FOUND'}"
        if c["status"] in ("MISMATCH", "WARN") and c.get("expected"):
            line += f"  (expected: {c['expected']})"
        print(line)
        if c["status"] in ("MISSING", "MISMATCH"):
            all_ok = False
    print(f"{'='*60}")
    print(f"Result: {'ALL OK' if all_ok else 'ISSUES FOUND'}")
    return all_ok


if __name__ == "__main__":
    report = check_environment()
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))
    else:
        ok = print_report(report)
        sys.exit(0 if ok else 1)
