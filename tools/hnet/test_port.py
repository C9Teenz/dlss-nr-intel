"""Check the PyTorch port (model.py) against the numpy oracle (reference nr_model.py).

    python tools/hnet/test_port.py                       # synthetic weights, 128x128 field
    python tools/hnet/test_port.py --weights real --size 320

1. primitives: exact ops bit-for-bit against the oracle's, deploy ops' error reported;
2. the full forward in exact mode, with the oracle's GEMMs redirected to the same
   torch.matmul, block by block: must agree to the bit (or within a hair, see --tol);
3. the full forward in each deploy numerics, against the exact head: the error the
   ONNX graph will carry, and the composed-image PSNR it costs.
Exit status 0 only when 1 and 2 pass.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402
import frame  # noqa: E402
from model import HNet, Numerics, Ops  # noqa: E402

torch.set_num_threads(8)
ref = common.import_reference_model(chunk_tokens=0)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def redirect_oracle_gemms():
    def mm(a, b):
        return torch.matmul(t(a), t(b)).numpy()

    def mm_nt(a, b):
        return torch.matmul(t(a), t(np.swapaxes(b, -1, -2))).numpy()

    ref.MATMUL, ref.MATMUL_NT = mm, mm_nt


def compare(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    diff = np.abs(a - b)
    scale = max(1e-12, float(np.abs(a).max()))
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else 1.0
    return dict(max=float(diff.max()), rel=float(diff.max() / scale),
                frac=float((diff > 0).mean()), corr=corr)


def test_primitives():
    print("primitives (exact vs oracle)")
    exact = Ops(Numerics.exact())
    rng = np.random.default_rng(1)
    every = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float32)
    every = every[np.isfinite(every)]
    wide = np.concatenate([every, rng.normal(0, 50, 100000).astype(np.float32)])
    check("e4m3 on every half value and 1e5 normals",
          np.array_equal(ref.e4m3(wide), exact.e4m3(t(wide)).numpy()))
    check("gate activation on every half value",
          np.array_equal(ref.quadratic_gate_activation(every), exact.gate_activation(t(every)).numpy()))
    scores = rng.normal(0, 3, (64, 2, 64, 64)).astype(np.float32)
    check("vendor window softmax",
          np.array_equal(ref.vendor_approximate_softmax(scores), exact.window_softmax(t(scores)).numpy()))
    vec = rng.normal(0, 4, (512, 2, 64, 32)).astype(np.float32)
    check("fragment-tree cosine normalise",
          np.array_equal(ref.vendor_cosine_normalize(vec), exact.cosine_normalize(t(vec)).numpy()))
    scale = rng.uniform(4, 9, 2).astype(np.float32)
    check("cosine publish with scale",
          np.array_equal(ref.vendor_cosine_publish(vec[:, :, :, :].transpose(0, 1, 2, 3), scale),
                         exact.cosine_publish(t(vec), t(scale)).numpy()))
    check("ViT exponential on every half value",
          np.array_equal(ref._vit_exp_weight(every), exact._vit_exp_weight(t(every)).numpy()))
    check("ViT norm", np.array_equal(ref._vit_norm(vec), exact.vit_norm(t(vec)).numpy()))
    q = ref.e4m3(rng.normal(0, 0.3, (1, 4, 100, 32)).astype(np.float32))
    k = ref.e4m3(rng.normal(0, 0.3, (1, 4, 100, 32)).astype(np.float32))
    v = ref.e4m3(rng.normal(0, 1, (1, 4, 100, 32)).astype(np.float32))
    check("ViT attend (padded keys, vendor denominator)",
          np.array_equal(ref.vit_attend(q, k, v), exact.vit_attend(t(q), t(k), t(v)).numpy()))

    print("primitives (deploy approximations vs exact)")
    for softmax in ("pwl", "exp"):
        ops = Ops(Numerics.deploy(e4m3="none", softmax=softmax))
        got = ops.window_softmax(t(scores)).numpy()
        want = ref.vendor_approximate_softmax(scores)
        print(f"    window softmax {softmax:<4}: max abs {np.abs(got - want).max():.4f} "
              f"(E4M3 quantum at 0.5 is 0.0625)")
        got = ops.vit_attend(t(q), t(k), t(v)).numpy()
        want = ref.vit_attend(q, k, v)
        print(f"    ViT attend     {softmax:<4}: {compare(want, got)}")
    ops = Ops(Numerics.deploy(e4m3="veltkamp"))
    sample = rng.normal(0, 30, 200000).astype(np.float32)
    agree = float((ops.e4m3(t(sample)).numpy() == ref.e4m3(sample)).mean())
    print(f"    veltkamp e4m3 (float32 carrier): {agree * 100:.3f} % identical to the exact rounding")
    ops16 = Ops(Numerics.deploy(e4m3="veltkamp"))
    sample16 = sample.astype(np.float16)
    got16 = ops16.e4m3(torch.from_numpy(sample16)).float().numpy()
    agree16 = float((got16 == ref.e4m3(sample16.astype(np.float32))).mean())
    print(f"    veltkamp e4m3 (half carrier):    {agree16 * 100:.3f} % identical to the exact rounding")


def oracle_forward(weights, features):
    """The oracle forward with every block's output captured, keyed as model.forward keys them."""
    captured = {}
    model = ref.NeuralRenderingModel(weights)
    original = {name: getattr(model, name) for name in
                ("_window", "_split_window", "_global", "_downsample_window", "_upsample_window")}

    def wrap(name):
        def run(value, *args, **kwargs):
            out = original[name](value, *args, **kwargs)
            index = args[1] if name == "_upsample_window" else args[0]
            captured[f"block{index}"] = out[1] if name == "_downsample_window" else out
            return out
        return run

    for name in original:
        setattr(model, name, wrap(name))
    head = model.forward(features[None])
    return head[0], captured


def test_forward(weights, size, tol):
    print(f"full forward, exact mode, {size}x{size} frame")
    redirect_oracle_gemms()
    color = frame.test_image(size, size)
    geometry = frame.network_geometry(size, size, minimum=frame.GRAPH_MINIMUM_EXTENT)
    features = frame.make_features(color, geometry)
    started = time.perf_counter()
    want, oracle_blocks = oracle_forward(weights, features)
    print(f"    oracle {time.perf_counter() - started:.1f} s", flush=True)
    model = HNet(weights, Numerics.exact())
    captured = {}
    started = time.perf_counter()
    with torch.no_grad():
        got = model(t(features)[None], capture=captured)[0].numpy()
    print(f"    port   {time.perf_counter() - started:.1f} s", flush=True)
    worst = None
    for name in sorted(oracle_blocks, key=lambda n: int(n[5:])):
        if name not in captured:
            continue
        stats = compare(oracle_blocks[name], captured[name].numpy()[0])
        if worst is None or stats["frac"] > worst[1]["frac"]:
            worst = (name, stats)
        if stats["frac"] > 0:
            print(f"    {name:<8} differs: {stats}")
    stats = compare(want, got)
    check("every block identical to the oracle" if tol == 0 else f"blocks within {tol}",
          worst[1]["frac"] <= tol, f"worst {worst[0]} {worst[1]}")
    check("head matches the oracle", stats["frac"] <= tol, str(stats))
    return features, color, geometry, want


def test_deploy(weights, features, color, geometry, exact_head):
    print("full forward, deploy numerics (float32 carrier) against the exact head")
    reference_out, _ = frame.compose(exact_head, color)
    for numerics in (Numerics.deploy("veltkamp", "pwl"), Numerics.deploy("none", "pwl"),
                     Numerics.deploy("veltkamp", "exp"), Numerics.deploy("none", "exp")):
        model = HNet(weights, numerics)
        with torch.no_grad():
            head = model(t(features)[None])[0].numpy()
        out, _ = frame.compose(head, color)
        stats = compare(exact_head, head)
        print(f"    e4m3={numerics.e4m3:<8} softmax={numerics.softmax:<4} head corr {stats['corr']:.4f} "
              f"max {stats['max']:.3f}  composed PSNR {frame.psnr(reference_out, out):.2f} dB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--size", type=int, default=128, help="square frame size (field follows)")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="largest fraction of differing values accepted per block")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()
    test_primitives()
    if not args.skip_forward:
        weights = common.get_weights(args.weights)
        features, color, geometry, head = test_forward(weights, args.size, args.tol)
        test_deploy(weights, features, color, geometry, head)
    print("FAILED: " + ", ".join(FAILURES) if FAILURES else "ALL PASS")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
