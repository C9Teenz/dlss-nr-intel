"""How close does our network get to the vendor's look? Runs the real-weight HNet (exact
numerics) on the 'DLSS 5 OFF' half of a side-by-side OFF|ON reference frame (1280x720, each half
640 wide) at several processing scales / profiles and compares the change it makes with the
change DLSS 5 makes (corr / gain / err as in --golden, plus a low/high-frequency split).

    python tools/hnet/compare_reference.py work/reference/dlss5_stellarblade_offon.png work/reference/out 0.3 0.5 1.0 1.5
    config: scale[:profile[:skin]]   e.g. 0.5  1.0:natural  1.0:standard:1

scale < 1 composes like the game bridge (bilinear head over the full-resolution frame);
scale >= 1 like MLX-DLSS (compose at the processing scale, resample back). The reference
stays under work/ (not ours to distribute). Results: PROGRESS-neural.md, 2026-11-05.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "hnet"))
import common  # noqa: E402
import frame  # noqa: E402
from model import HNet, Numerics  # noqa: E402

SHIFT = 17          # ON(x) ~= OFF(x + SHIFT) for the Stellar Blade reference (phase correlation)
TOP, BOTTOM = 120, 680   # skip the badges and the copyright line


def halves(path):
    im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    off = im[:, :640]
    on = im[:, 640:]
    return off, on


def resize(img, w, h):
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
    return np.clip(cv2.resize(img, (w, h), interpolation=interp), 0, 1).astype(np.float32)


def run(net, off, scale, profile, skin):
    h0, w0 = off.shape[:2]
    w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    proc = off if scale == 1 else resize(off, w, h)
    geometry = frame.network_geometry(w, h)
    feats = frame.make_features(proc, geometry, profile=profile)
    if skin is not None:   # MLX AutomaticMask(skin, skin): structure 1, skin + automatic strength
        feats[..., 12] = 1.0
        feats[..., 13] = skin
        feats[..., 14] = skin
    t = time.time()
    with torch.no_grad():
        head = net.forward(torch.from_numpy(feats[None]))[0].float().numpy()
    dt = time.time() - t
    head = head[:h, :w]
    if scale < 1:   # game path: bilinear head over the full-resolution frame
        up = cv2.resize(head, (w0, h0), interpolation=cv2.INTER_LINEAR)
        out, _ = frame.compose(up, off)
    else:           # MLX path: compose at the processing scale, resample back
        comp, _ = frame.compose(head, proc)
        out = comp if scale == 1 else resize(comp, w0, h0)
    return out, head, dt, (geometry.field_width, geometry.field_height)


def metrics(off, on, out):
    a = slice(TOP, BOTTOM)
    o, r, e = off[a, SHIFT:], on[a, : 640 - SHIFT], out[a, SHIFT:]
    dr, de = (r - o).ravel(), (e - o).ravel()
    corr = float(np.corrcoef(dr, de)[0, 1])
    gain = float(de @ dr / (dr @ dr))
    err = float(np.sqrt(np.mean((de - dr) ** 2)) / np.sqrt(np.mean(dr ** 2)))
    # split low (sigma 6) / high frequency of the change
    def split(x):
        low = cv2.GaussianBlur(x, (0, 0), 6)
        return low, x - low
    (dl, dh), (el, eh) = split(r - o), split(e - o)
    c_low = float(np.corrcoef(dl.ravel(), el.ravel())[0, 1])
    c_high = float(np.corrcoef(dh.ravel(), eh.ravel())[0, 1])
    g_high = float((eh.ravel() @ dh.ravel()) / (dh.ravel() @ dh.ravel()))
    return dict(psnr_on=frame.psnr(e, r), psnr_off_on=frame.psnr(o, r), corr=corr, gain=gain, err=err,
                corr_low=c_low, corr_high=c_high, gain_high=g_high,
                mean_shift=(e - o).mean((0, 1)), ref_shift=(r - o).mean((0, 1)))


def main():
    ref, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    off, on = halves(ref)
    net = HNet(common.get_weights("real"), Numerics.exact())
    net.eval()
    print(f"baseline OFF vs ON: {metrics(off, on, off)['psnr_off_on']:.2f} dB", flush=True)
    for cfg in sys.argv[3:]:
        parts = cfg.split(":")
        scale = float(parts[0])
        profile = parts[1] if len(parts) > 1 else "standard"
        skin = float(parts[2]) if len(parts) > 2 else None
        out, head, dt, field = run(net, off, scale, profile, skin)
        m = metrics(off, on, out)
        name = f"s{scale}_{profile}" + (f"_skin{skin}" if skin is not None else "")
        frame.save_image(out_dir / f"{name}.png", out)
        np.save(out_dir / f"{name}_head.npy", head.astype(np.float16))
        print(f"{name:28s} field {field[0]}x{field[1]} {dt:6.1f}s | PSNR vs ON {m['psnr_on']:.2f} (OFF {m['psnr_off_on']:.2f}) "
              f"| corr {m['corr']:.3f} gain {m['gain']:.3f} err {m['err']:.3f} | low {m['corr_low']:.3f} "
              f"high {m['corr_high']:.3f} (gain {m['gain_high']:.3f}) | mean d {np.round(m['mean_shift'], 4)} "
              f"ref {np.round(m['ref_shift'], 4)} | head rms {float(np.sqrt((head[..., :3] ** 2).mean())):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
