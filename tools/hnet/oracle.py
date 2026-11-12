"""N0 oracle: run the numpy reference (nr_model, exact vendor numerics) on test frames and
keep its output as the golden reference every later stage is judged against.

    python tools/hnet/oracle.py                         # real weights, default frames
    python tools/hnet/oracle.py --image shot.png --frames 320x320 960x540
    python tools/hnet/oracle.py --weights synthetic     # plumbing check without the DLL

Writes work/golden/<name>_<W>x<H>.npz with
    color     (H, W, 3)        the frame (float32, [0, 1])
    features  (fh, fw, 16)     the network input on the padded field
    head      (fh, fw, 4)      the network output (uncropped)
    output    (H, W, 3)        the composed first frame (no history)
    history_output (H, W, 3)   a second frame composed with the first one's prediction as
                               history (features 7-9 and the temporal gate exercised)
    blocks/<name>              per-block outputs (with --blocks; large)
a PNG of `output` and <stem>_{color,output,history_output}.pfm beside it. Everything under work/ stays local.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import frame  # noqa: E402

DEFAULT_FRAMES = ("320x320", "960x540")


def run_oracle(model, ref, features, capture_blocks):
    blocks = {}
    if capture_blocks:
        names = ("_window", "_split_window", "_global", "_downsample_window", "_upsample_window")
        original = {name: getattr(model, name) for name in names}

        def wrap(name):
            def run(value, *args, **kwargs):
                out = original[name](value, *args, **kwargs)
                index = args[1] if name == "_upsample_window" else args[0]
                blocks[f"block{index}"] = (out[1] if name == "_downsample_window" else out)[0].astype(np.float16)
                return out
            return run

        for name in names:
            setattr(model, name, wrap(name))
    head = model.forward(features[None])[0]
    return head, blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", choices=("real", "synthetic"), default="real")
    parser.add_argument("--image", type=Path, help="source picture (resized to each frame size)")
    parser.add_argument("--frames", nargs="+", default=list(DEFAULT_FRAMES))
    parser.add_argument("--blocks", action="store_true", help="also dump every block's output")
    parser.add_argument("--chunk-tokens", type=int, default=1 << 13)
    args = parser.parse_args()

    ref = common.import_reference_model(chunk_tokens=args.chunk_tokens)
    weights = common.get_weights(args.weights)
    model = ref.NeuralRenderingModel(weights)
    out_dir = common.GOLDEN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "synw_" if args.weights == "synthetic" else ""
    source = frame.load_image(args.image) if args.image else None

    for text in args.frames:
        width, height = (int(v) for v in text.split("x"))
        if source is not None:
            from PIL import Image
            img = Image.fromarray((source * 255 + 0.5).astype(np.uint8)).resize((width, height), Image.LANCZOS)
            color = np.asarray(img, np.float32) / 255.0
        else:
            color = frame.test_image(width, height)
        geometry = frame.network_geometry(width, height)
        features = frame.make_features(color, geometry)
        started = time.perf_counter()
        head, blocks = run_oracle(model, ref, features, args.blocks)
        output, prediction = frame.compose(head, color)
        print(f"{text}: field {geometry.field_width}x{geometry.field_height}, "
              f"oracle {time.perf_counter() - started:.1f} s", flush=True)

        # second frame: the same picture with the first prediction as history
        # frame index 0 again: the live mode never advances it (the noise stays put)
        features2 = frame.make_features(color, geometry, history=prediction, frame_index=0)
        head2, _ = run_oracle(model, ref, features2, False)
        history_output, _ = frame.compose(head2, color, history=prediction)

        stem = out_dir / f"{tag}{'image' if source is not None else 'testimg'}_{width}x{height}"
        payload = dict(color=color, features=features.astype(np.float32), head=head.astype(np.float32),
                       output=output, features2=features2.astype(np.float32),
                       head2=head2.astype(np.float32), history_output=history_output)
        payload.update({f"blocks/{k}": v for k, v in blocks.items()})
        np.savez_compressed(stem.with_suffix(".npz"), **payload)
        frame.save_image(stem.with_suffix(".png"), output)
        for key in ("color", "output", "history_output"):        # for dlss_nr_dx12_poc --neural --golden
            frame.save_pfm(f"{stem}_{key}.pfm", payload[key])
        print(f"  saved {stem}.npz; |residual| mean {np.abs(output - color).mean() * 255:.2f} levels, "
              f"PSNR(out, in) {frame.psnr(output, color):.2f} dB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
