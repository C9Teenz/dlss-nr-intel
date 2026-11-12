"""(Re)make every model artifact the C++ runner, its checks and the game drop-in use.

    python tools/hnet/make_models.py --weights synthetic        # stand-ins, no NVIDIA data
    python tools/hnet/make_models.py --weights real             # from work/mlxw (your DLL's weights)
    python tools/hnet/make_models.py --weights real --fields 320x320 960x576 1344x768 --full

Always: the weight blob the native runner needs (work/onnx/hnet-ends_nq-exp[_synthetic].bin),
the reference dumps (work/hybrid/<first field>_{deep,shallow}[...]) and the oracle golden of the
first field (work/golden). Per field: the hybrid middles (only needed without the native path)
and, at the first field or with --full, the whole-graph ONNX (nq-exp; vk-pwl at the first field
for the golden check of the ONNX path).

Made from real weights, every file here embeds them: they stay in work/ and are never shared.
Afterwards: python tools/run_neural_checks.py [--real]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("+ " + " ".join(cmd[1:]), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if subprocess.run(cmd, env=env).returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", choices=("real", "synthetic"), required=True)
    parser.add_argument("--fields", nargs="+", default=["320x320", "960x576"])
    parser.add_argument("--full", action="store_true", help="the whole-graph ONNX at every field")
    parser.add_argument("--no-onnx", action="store_true", help="only the blob, dumps and golden (native path)")
    args = parser.parse_args()
    weights = ["--weights", args.weights]
    export = HERE / "export_onnx.py"
    deploy = [*weights, "--e4m3", "none", "--softmax", "exp"]
    if not args.no_onnx:
        for i, field in enumerate(args.fields):
            run(export, "--field", field, *deploy, "--part", "middle-deep")
            run(export, "--field", field, *deploy, "--part", "middle")
            if args.full or i == 0:
                run(export, "--field", field, *deploy)
            if i == 0:
                run(export, "--field", field, *weights)          # vk-pwl, the ONNX path's golden check
    first = args.fields[0]
    run(HERE / "export_hybrid.py", *weights, "--dump", first, "--split", "deep")
    run(HERE / "export_hybrid.py", *weights, "--dump", first, "--split", "shallow")
    run(HERE / "oracle.py", *weights, "--frames", first)


if __name__ == "__main__":
    main()
