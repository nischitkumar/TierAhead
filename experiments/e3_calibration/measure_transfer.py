#!/usr/bin/env python3
"""E3 step 2 -- Host->device pinned-memory transfer bandwidth (GPU-only;
not run on the machine that wrote this file -- see E3_CALIBRATION.md's
"what was and wasn't run" section).

Measures real H2D bandwidth vs transfer size using the same CUDA-event
pattern `e1_roofline/measure_compute.py`'s LayerTimer already uses (record
start/end events around the operation, synchronize once, read back elapsed
time -- avoids CPU-side timer overhead contaminating a GPU-side
measurement). transfer_fit.py (run separately, no GPU needed) fits the
result to Experiments.md E3 step 2's `t = alpha + size/beta` model.

Usage (on the GPU server):
  python3 measure_transfer.py --out out/calib_transfer.json
"""
import argparse
import json
import statistics
from pathlib import Path

import torch


def bench_one_size(size_bytes: int, warmup: int, reps: int, device: str):
    """fp32 elements so size_bytes is exactly reproducible (4 bytes/elem,
    no fp16/bf16 rounding of the requested byte count)."""
    n_elements = max(1, size_bytes // 4)
    cpu_pinned = torch.empty(n_elements, dtype=torch.float32, pin_memory=True)
    cpu_pinned.uniform_()
    gpu_buf = torch.empty(n_elements, dtype=torch.float32, device=device)

    for _ in range(warmup):
        gpu_buf.copy_(cpu_pinned, non_blocking=True)
    torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        gpu_buf.copy_(cpu_pinned, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes-mb", default="1,2,4,8,16,32,64,128,256,512",
                     help="Experiments.md E3 step 2: '1 MB -> 512 MB'")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out/calib_transfer.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available -- check nvidia-smi/driver/torch install"

    sizes_mb = [float(x) for x in args.sizes_mb.split(",")]
    by_size = {}
    for mb in sizes_mb:
        size_bytes = int(mb * 1e6)
        print(f"[measure_transfer] size={mb:g}MB ({size_bytes} bytes): warmup={args.warmup} reps={args.reps}",
              flush=True)
        samples = bench_one_size(size_bytes, args.warmup, args.reps, args.device)
        mean_ms = float(statistics.mean(samples))
        by_size[str(size_bytes)] = {
            "size_mb": mb,
            "mean_ms": mean_ms,
            "p50_ms": float(statistics.median(samples)),
            "p95_ms": float(sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]),
            "n_samples": len(samples),
        }
        achieved_gbps = (size_bytes / 1e9) / (mean_ms / 1e3) if mean_ms > 0 else float("inf")
        print(f"[measure_transfer]   mean={mean_ms:.4f} ms ({achieved_gbps:.2f} GB/s achieved)", flush=True)

    entry = {
        "provenance": "measured_here",
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "device": args.device,
        "warmup": args.warmup,
        "reps": args.reps,
        "by_size_bytes": by_size,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entry, indent=2))
    print(f"[measure_transfer] DONE. wrote {out_path}", flush=True)
    print("[measure_transfer] next: python3 build_calibration_report.py (fits alpha/beta, no GPU needed)",
          flush=True)


if __name__ == "__main__":
    main()
