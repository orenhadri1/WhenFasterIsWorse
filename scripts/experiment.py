# experiment_full.py
# Full, self-contained experiment for Windows + NVIDIA GPU (WDDM) with:
# - Fixed precision (FP32)
# - Identical "work" across policies (same N matmuls of same size)
# - Policy changes only via scheduling: chunking + CPU sleep between chunks
# - Power logging via nvidia-smi (read-only)
# - Energy computed by trapezoidal integration of sampled power
#
# Usage (PowerShell/CMD):
#   python experiment_full.py --out results.csv
#
# Notes:
# - Requires: Python + torch + pandas + numpy
# - Ensure nvidia-smi is in PATH (it is if NVIDIA driver is installed)


# Setup env: 
#     python3 -m venv venv
#     source venv/bin/activate
#     pip install --upgrade pip
#     pip install torch --index-url https://download.pytorch.org/whl/cu121
#     pip install 'numpy>=1.23'
# command line: 
#     python experiment_full.py --out results.csv --reps 6 --N 2000 --size 2048


import argparse
import os
import sys
import time
import subprocess
import signal
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class Policy:
    name: str
    N: int
    size: int
    chunk: int
    sleep_ms: float
    warmup: int


@dataclass
class RunResult:
    run_id: str
    policy: str
    rep: int
    N: int
    size: int
    chunk: int
    sleep_ms: float
    total_time_s: float
    samples: int
    mean_power_w: float
    energy_j: float
    joules_per_iter: float
    start_ts: str
    end_ts: str
    notes: str


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch build and NVIDIA driver.")
    _ = torch.cuda.get_device_name(0)


def build_fixed_workload(size: int, dtype=torch.float32):
    # Keep tensors on GPU to avoid I/O and CPU->GPU transfer noise.
    device = "cuda"
    x = torch.randn(size, size, device=device, dtype=dtype)
    w = torch.randn(size, size, device=device, dtype=dtype)
    return x, w


def do_work_fp32(x: torch.Tensor, w: torch.Tensor, iters: int) -> torch.Tensor:
    # Deterministic sequence of GPU kernels: repeated GEMM.
    # Same number of GEMMs == same defined "work".
    for _ in range(iters):
        x = x @ w
    return x


def start_power_logger(csv_path: str, interval_ms: int = 100) -> subprocess.Popen:
    """
    Starts nvidia-smi sampling in a background process writing CSV.
    We log timestamp and power.draw for energy integration, plus clocks/util/temp for diagnostics.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,power.draw,utilization.gpu,clocks.gr,temperature.gpu",
        "--format=csv,noheader,nounits",
        "-lms",
        str(interval_ms),
    ]

    # Use CREATE_NEW_PROCESS_GROUP so we can terminate cleanly on Windows
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    # Open file handle for stdout
    f = open(csv_path, "w", encoding="utf-8", newline="")
    try:
        p = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            text=True,
        )
    except Exception:
        f.close()
        raise
    # Attach file handle to process object so caller can close it after termination
    p._power_log_file = f  # type: ignore[attr-defined]
    return p


def stop_power_logger(p: subprocess.Popen) -> None:
    # Terminate logger process and close its file.
    if p.poll() is None:
        try:
            if os.name == "nt":
                # CTRL_BREAK_EVENT requires process group; this is best-effort.
                p.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                time.sleep(0.2)
            p.terminate()
            try:
                p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                p.kill()
        except Exception:
            # Final fallback
            try:
                p.kill()
            except Exception:
                pass

    f = getattr(p, "_power_log_file", None)
    if f is not None:
        try:
            f.flush()
            f.close()
        except Exception:
            pass


def parse_power_log(csv_path: str) -> pd.DataFrame:
    # Expected columns: timestamp, index, power.draw, utilization.gpu, clocks.gr, temperature.gpu
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 3:
        raise RuntimeError(f"Power log file looks malformed: {csv_path}")

    df = df.iloc[:, :6]
    df.columns = ["timestamp", "gpu", "power_w", "util", "clocks_gr", "temp_c"]

    # Convert numeric columns
    for c in ["gpu", "power_w", "util", "clocks_gr", "temp_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["power_w"]).reset_index(drop=True)

    # Parse timestamps if possible (nvidia-smi format like: "2025/12/28 18:52:58.123")
    # If parsing fails, we fall back to constant dt assumption later.
    try:
        df["ts_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception:
        df["ts_parsed"] = pd.NaT

    return df


def integrate_energy_j(df: pd.DataFrame, default_dt_s: float = 0.1) -> Tuple[float, float, int, str, str]:
    """
    Returns:
      energy_j, mean_power_w, samples, start_ts, end_ts
    """
    if len(df) < 3:
        return 0.0, float(df["power_w"].mean()) if len(df) else 0.0, len(df), "", ""

    p = df["power_w"].to_numpy(dtype=float)

    # Use timestamp-derived dt if valid; else assume constant dt.
    if "ts_parsed" in df.columns and df["ts_parsed"].notna().all():
        t = df["ts_parsed"].astype("int64").to_numpy(dtype=float) / 1e9  # seconds
        dt = np.diff(t)
        # Guard: remove non-positive or absurd dt values (logger jitter)
        dt = np.clip(dt, 0.001, 1.0)
        energy = float(np.sum((p[:-1] + p[1:]) * 0.5 * dt))
        start_ts = str(df["ts_parsed"].iloc[0])
        end_ts = str(df["ts_parsed"].iloc[-1])
    else:
        dt = default_dt_s
        energy = float(np.sum((p[:-1] + p[1:]) * 0.5 * dt))
        start_ts = str(df["timestamp"].iloc[0]) if "timestamp" in df.columns else ""
        end_ts = str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else ""

    mean_power = float(np.mean(p))
    return energy, mean_power, len(p), start_ts, end_ts


def run_policy_once(policy: Policy, run_id: str, rep: int, log_dir: str, interval_ms: int = 100) -> RunResult:
    # Build fixed tensors once per run to keep memory footprint stable.
    # (Still same work definition across policies: N GEMMs.)
    x, w = build_fixed_workload(policy.size, dtype=torch.float32)

    # Warmup (same count for all policies)
    for _ in range(policy.warmup):
        x = x @ w
    torch.cuda.synchronize()

    log_path = os.path.join(log_dir, f"power_{run_id}_{policy.name}_rep{rep}.csv")
    logger = start_power_logger(log_path, interval_ms=interval_ms)

    notes = ""
    try:
        t0 = time.time()

        done = 0
        while done < policy.N:
            this = min(policy.chunk, policy.N - done)
            x = do_work_fp32(x, w, this)
            torch.cuda.synchronize()
            done += this

            if done < policy.N and policy.sleep_ms > 0:
                time.sleep(policy.sleep_ms / 1000.0)

        t1 = time.time()
        total_time = t1 - t0

    except Exception as e:
        total_time = float("nan")
        notes = f"Exception during work loop: {e}"
    finally:
        stop_power_logger(logger)

    df = parse_power_log(log_path)
    energy_j, mean_power_w, samples, start_ts, end_ts = integrate_energy_j(df, default_dt_s=interval_ms / 1000.0)

    j_per_iter = energy_j / policy.N if policy.N > 0 else float("nan")

    return RunResult(
        run_id=run_id,
        policy=policy.name,
        rep=rep,
        N=policy.N,
        size=policy.size,
        chunk=policy.chunk,
        sleep_ms=policy.sleep_ms,
        total_time_s=float(total_time),
        samples=int(samples),
        mean_power_w=float(mean_power_w),
        energy_j=float(energy_j),
        joules_per_iter=float(j_per_iter),
        start_ts=start_ts,
        end_ts=end_ts,
        notes=notes,
    )


def interleave_runs(policies: List[Policy], reps: int) -> List[Tuple[Policy, int]]:
    # Order: A1, B1, C1, ... A2, B2, ...
    schedule = []
    for r in range(1, reps + 1):
        for p in policies:
            schedule.append((p, r))
    return schedule


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    # Summary per policy: median time, median energy, median power, IQRs
    def iqr(x):
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    grp = df.groupby("policy", as_index=False).agg(
        runs=("policy", "count"),
        time_median=("total_time_s", "median"),
        time_iqr=("total_time_s", iqr),
        energy_median=("energy_j", "median"),
        energy_iqr=("energy_j", iqr),
        power_median=("mean_power_w", "median"),
        power_iqr=("mean_power_w", iqr),
        j_per_iter_median=("joules_per_iter", "median"),
        j_per_iter_iqr=("joules_per_iter", iqr),
    )

    # Identify "time-optimal" and "energy-optimal" policies (by medians)
    time_opt = grp.loc[grp["time_median"].idxmin(), "policy"]
    energy_opt = grp.loc[grp["energy_median"].idxmin(), "policy"]
    grp["is_time_optimal"] = grp["policy"].eq(time_opt)
    grp["is_energy_optimal"] = grp["policy"].eq(energy_opt)
    return grp.sort_values("time_median").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.csv", help="Output CSV with per-run results")
    ap.add_argument("--log-dir", default="logs", help="Directory for raw power logs")
    ap.add_argument("--reps", type=int, default=6, help="Repetitions per policy (interleaved)")
    ap.add_argument("--interval-ms", type=int, default=100, help="nvidia-smi sampling interval (ms)")
    ap.add_argument("--N", type=int, default=2000, help="Total GEMM iterations (work) per run")
    ap.add_argument("--size", type=int, default=2048, help="Matrix size (reduce if OOM)")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup iterations before measurement")
    args = ap.parse_args()

    ensure_cuda()
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}")

    # Policies: same work (N,size,warmup,FP32), only scheduling differs.
    # A: continuous (time-optimal baseline)
    # B/C/D: chunked with increasing sleep (candidate energy-optimal)
    policies = [
        Policy(name="A_continuous", N=args.N, size=args.size, chunk=args.N, sleep_ms=0.0, warmup=args.warmup),
        Policy(name="B_chunk100_sleep1", N=args.N, size=args.size, chunk=100, sleep_ms=1.0, warmup=args.warmup),
        Policy(name="C_chunk100_sleep3", N=args.N, size=args.size, chunk=100, sleep_ms=3.0, warmup=args.warmup),
        Policy(name="D_chunk100_sleep6", N=args.N, size=args.size, chunk=100, sleep_ms=6.0, warmup=args.warmup),
    ]

    # Optional: add a "different scheduling shape" with larger chunks (same N)
    policies += [
        Policy(name="E_chunk400_sleep2", N=args.N, size=args.size, chunk=400, sleep_ms=2.0, warmup=args.warmup),
    ]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)

    schedule = interleave_runs(policies, args.reps)

    rows: List[RunResult] = []
    print(f"Run ID: {run_id}")
    print(f"Total runs: {len(schedule)} (policies={len(policies)} x reps={args.reps})")

    for idx, (pol, rep) in enumerate(schedule, start=1):
        print(f"[{idx}/{len(schedule)}] Running {pol.name} rep {rep} ...")
        res = run_policy_once(pol, run_id=run_id, rep=rep, log_dir=args.log_dir, interval_ms=args.interval_ms)
        rows.append(res)
        print(f"  T={res.total_time_s:.3f}s  meanP={res.mean_power_w:.1f}W  E={res.energy_j:.0f}J")

        # Short cool-down to reduce thermal drift between interleaved runs
        time.sleep(3.0)

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(args.out, index=False)

    summary = summarize_results(df)
    summary_path = os.path.splitext(args.out)[0] + "_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Print key claim for the paper:
    # "time-optimal policy is not energy-optimal"
    time_opt = summary.loc[summary["is_time_optimal"], "policy"].iloc[0]
    energy_opt = summary.loc[summary["is_energy_optimal"], "policy"].iloc[0]

    t_time_opt = summary.loc[summary["policy"] == time_opt, "time_median"].iloc[0]
    e_time_opt = summary.loc[summary["policy"] == time_opt, "energy_median"].iloc[0]

    t_energy_opt = summary.loc[summary["policy"] == energy_opt, "time_median"].iloc[0]
    e_energy_opt = summary.loc[summary["policy"] == energy_opt, "energy_median"].iloc[0]

    print("\nSummary written to:", summary_path)
    print("Time-optimal policy:", time_opt, f"(median T={t_time_opt:.3f}s, median E={e_time_opt:.0f}J)")
    print("Energy-optimal policy:", energy_opt, f"(median T={t_energy_opt:.3f}s, median E={e_energy_opt:.0f}J)")

    if time_opt != energy_opt:
        print("\nDecision inversion observed:")
        print(f"- Minimizing time selects {time_opt}")
        print(f"- Minimizing energy selects {energy_opt}")
    else:
        print("\nNo inversion observed with current policy set.")
        print("Increase policy diversity (more chunk/sleep points) and/or increase N to reduce noise.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
