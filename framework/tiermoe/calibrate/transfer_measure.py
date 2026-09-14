"""Host->device pinned-memory transfer bandwidth (GPU-required). Measures
real H2D bandwidth vs transfer size using paired CUDA events (record start/
end, synchronize once, read back elapsed time). transfer_fit.py (no GPU
needed) fits the result to t = alpha + size/beta.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def bench_one_size(size_bytes: int, warmup: int, reps: int, device: str):
    import torch

    n_elements = max(1, size_bytes // 4)
    cpu_pinned = torch.empty(n_elements, dtype=torch.float32, pin_memory=True)
    cpu_pinned.uniform_()
    gpu_buf = torch.empty(n_elements, dtype=torch.float32, device=device)
    for _ in range(warmup):
        gpu_buf.copy_(cpu_pinned, non_blocking=True)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        gpu_buf.copy_(cpu_pinned, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main(argv=None):
    from tiermoe.hw import probe

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes-mb", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="calib_transfer.json")
    args = ap.parse_args(argv)

    if not probe().can_run_real_gpu_calibration:
        print("tiermoe calibrate transfer_measure: refusing -- no CUDA on this machine.", file=sys.stderr)
        sys.exit(1)

    import torch

    by_size = {}
    for mb in [float(x) for x in args.sizes_mb.split(",")]:
        size_bytes = int(mb * 1e6)
        samples = bench_one_size(size_bytes, args.warmup, args.reps, args.device)
        mean_ms = float(statistics.mean(samples))
        by_size[str(size_bytes)] = {"size_mb": mb, "mean_ms": mean_ms,
                                     "p50_ms": float(statistics.median(samples)),
                                     "p95_ms": float(sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]),
                                     "n_samples": len(samples)}
        print(f"[calibrate.transfer_measure] {mb:g}MB: mean={mean_ms:.4f}ms "
              f"({(size_bytes/1e9)/(mean_ms/1e3):.2f} GB/s)")

    entry = {"provenance": "measured_here", "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "device": args.device, "warmup": args.warmup, "reps": args.reps, "by_size_bytes": by_size}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entry, indent=2))
    print(f"[calibrate.transfer_measure] wrote {out_path}")


if __name__ == "__main__":
    main()
