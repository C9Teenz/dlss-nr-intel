"""Every check of the neural port in one go (synthetic weights unless --real).

    python tools/run_neural_checks.py            # after cmake --build build and make_models.py --weights synthetic
    python tools/run_neural_checks.py --real     # after make_models.py --weights real

1. the PyTorch port against the numpy reference (bit-exact, all 71 blocks)
2. the hybrid kernels against PyTorch: shallow split, and native (front, core, back)
3. the C++ graph against the oracle golden: native, and the whole-graph ONNX (vk-pwl)
4. the game bridge on a real swapchain (resize, 60 frames)
Exit status 0 only when all pass.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "build" / "dlss_nr_dx12_poc.exe"
BRIDGE = ROOT / "build" / "test_bridge.exe"


def run(label, cmd, must_contain):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    out = result.stdout + result.stderr
    ok = result.returncode == 0 and must_contain in out
    detail = [line.strip() for line in out.splitlines() if "PSNR" in line or "->" in line or "FAIL" in line or "effect frame" in line
              or "flicker" in line][:6]
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + "".join(f"\n         {d}" for d in detail), flush=True)
    if not ok and not detail:
        print("\n".join("         " + line for line in out.splitlines()[-15:]))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    weights = [] if args.real else ["--synthetic"]
    prefix = "" if args.real else "synw_"
    suffix = "" if args.real else "_synthetic"
    py = sys.executable
    results = [
        run("port vs numpy reference", [py, "tools/hnet/test_port.py", "--size", "128"]
            + (["--weights", "real"] if args.real else []), "ALL PASS"),
        run("hybrid shallow vs PyTorch", [APP, "--neural", *weights, "--hybrid-check",
                                          f"work/hybrid/320x320_shallow{suffix}", "--output-dir", "work"], "-> PASS"),
        run("native (front, core, back) vs PyTorch", [APP, "--neural", *weights, "--hybrid-check",
                                                      f"work/hybrid/320x320_deep{suffix}", "--output-dir", "work"],
            "-> PASS"),
        run("native vs oracle golden", [APP, "--neural", *weights, "--golden",
                                        f"work/golden/{prefix}testimg_320x320", "--output-dir", "work"], "-> PASS"),
        run("native over two frames (interval 2): parts consistent", [APP, "--neural", *weights, "--interval", "2",
                                                                       "--no-temporal", "--golden",
                                                                       f"work/golden/{prefix}testimg_320x320",
                                                                       "--output-dir", "work"],
            "interval 2, frames 3-6 vs frame 1: max |d| 0.0000  -> PASS"),
        run("native over four frames (interval 4): parts consistent, still warp exact",
            [APP, "--neural", *weights, "--interval", "4", "--no-temporal", "--golden",
             f"work/golden/{prefix}testimg_320x320", "--output-dir", "work"],
            "interval 4, frames 3-10 vs frame 1: max |d| 0.0000  -> PASS"),
        run("native, precision fast (no E4M3 roundings) vs oracle golden",
            [APP, "--neural", *weights, "--precision", "fast", "--golden", f"work/golden/{prefix}testimg_320x320",
             "--output-dir", "work"], "-> PASS"),
        run("head stabiliser (precision accurate): still calm, motion not dragged",
            [APP, "--neural", *weights, "--precision", "accurate", "--no-temporal", "--stabilize", "0.05",
             "--stability", f"work/golden/{prefix}testimg_320x320", "--output-dir", "work"], "stabiliser -> PASS"),
        run("whole-graph ONNX vk-pwl vs oracle golden", [APP, "--neural", *weights, "--tag", "vk-pwl", "--golden",
                                                         f"work/golden/{prefix}testimg_320x320", "--output-dir", "work"],
            "-> PASS"),
    ]
    if args.real:   # the synthetic network's correction hardly depends on the picture: nothing to follow
        for n in (4, 2):
            results.append(run(f"interval {n} warp (precision fast): beats the unmoved head on pans, an object, "
                               "non-rigid motion, a rotation, a cut; pans jump no more than at interval 1",
                               [APP, "--neural", "--precision", "fast", "--no-temporal", "--interval", str(n),
                                "--interval-check", "work/golden/testimg_320x320", "--output-dir", "work"],
                               "warp -> PASS"))
    if not args.real:
        results.append(run("game bridge (swapchain, resize)", [BRIDGE, "--output-dir", "work", "--frames", "60"],
                           "PASSED"))
        results.append(run("game bridge, BGRA swapchain, background setup",
                           [BRIDGE, "--bgra", "--async", "--output-dir", "work", "--frames", "240"], "PASSED"))
        results.append(run("game bridge, network over two frames (interval 2)",
                           [BRIDGE, "--interval", "2", "--output-dir", "work", "--frames", "60"], "PASSED"))
        results.append(run("game bridge, network over four frames (interval 4, warp), ini saved mid-run reloaded",
                           [BRIDGE, "--interval", "4", "--reload", "--output-dir", "work", "--frames", "60"], "PASSED"))
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
