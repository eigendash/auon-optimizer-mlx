#!/usr/bin/env python
"""Compare AuON, Hybrid-AuON, Muon, SGD-momentum and Adam on a toy induction task.

The script has four parts, all logged:

1. ``tune``    -- a coarse learning-rate sweep per arm on seed 0, because the
                  paper tunes each optimizer separately (its section 4.1 uses
                  0.24 for AuON and 0.01 for Muon, which is not a like-for-like
                  step size).
2. ``main``    -- the same model, task, batch stream and seeds for every arm,
                  with the tuned learning rate; reports final loss and next-token
                  accuracy, the peak and final largest causal attention logit,
                  wall-clock and the optimizer's own share of it.
3. ``scaling`` -- seconds per optimizer step for one square matrix as the side
                  grows, plus the analytic multiply-accumulate proxy; the
                  linear-time claim is about this curve.
4. ``brake``   -- a controlled spike sweep through the optimizer itself, and a
                  10x learning-rate training stress test.

Everything is CPU-only (``mlx.core.set_default_device(mx.cpu)``): the GPU
embedding backward pass is not bit-reproducible, and the timing curve would be
dominated by launch overhead at these sizes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auon.data import InductionTask, next_token_accuracy  # noqa: E402
from auon.normalizers import spectral_norm  # noqa: E402
from auon.optim import (  # noqa: E402
    TEMPERATURE_SCALED,
    VANILLA,
    AuON,
    HybridAuON,
    Muon,
)
from auon.scaling import analytic_macs, loglog_slope, scaling_sweep  # noqa: E402
from auon.train import TrainConfig, make_model, train_arm  # noqa: E402

ARMS = ["adam", "sgd_momentum", "muon", "auon", "auon_temp", "hybrid_auon"]

#: One order of magnitude either side of the values that looked sane in a pilot.
LR_GRID = {
    "adam": [3e-4, 1e-3, 3e-3],
    "sgd_momentum": [3e-3, 1e-2, 3e-2],
    "muon": [2e-3, 5e-3, 2e-2],
    "auon": [5e-3, 2e-2, 5e-2],
    "auon_temp": [2e-2, 5e-2, 2e-1],
    "hybrid_auon": [5e-3, 2e-2, 5e-2],
}

SCALING_ARMS = [
    "adam",
    "sgd_momentum",
    "auon",
    "auon_temp",
    "hybrid_auon_ns1",
    "muon_ns1",
    "muon",
    "hybrid_auon",
]
SCALING_SIZES = [128, 256, 512, 1024, 2048, 4096]


class Log:
    """Print and keep everything the script says, for results/."""

    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = []

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        self.lines.append(message)

    def save(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n")


def mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "values": []}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "values": [float(v) for v in values],
    }


def batch_stream(task, seed: int, steps: int, batch_size: int):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(steps):
        yield task.batch(rng, batch_size)


def build_task(args) -> InductionTask:
    return InductionTask(vocab_size=args.vocab_size, repeat_len=args.repeat_len)


def model_kwargs(args) -> dict:
    return dict(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    )


def train_config(args, learning_rate: float, steps: int, eval_every: int) -> TrainConfig:
    return TrainConfig(
        steps=steps,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        adam_learning_rate=args.adam_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rho=args.rho,
        ns_steps=args.ns_steps,
        eval_every=eval_every,
        log_every=0,
    )


def run_arm(args, task, arm: str, learning_rate: float, seed: int, steps: int, eval_every: int, log_every: int = 0):
    model = make_model(seed, vocab_size=task.n_tokens, **model_kwargs(args))
    config = train_config(args, learning_rate, steps, eval_every)
    config.log_every = log_every
    return train_arm(
        arm,
        model,
        batch_stream(task, seed, steps, args.batch_size),
        task.fixed_eval(args.eval_sequences),
        config,
    )


def thin(values: list, every: int = 5) -> list:
    thinned = [float(v) for v in values[::every]]
    if values and float(values[-1]) != thinned[-1]:
        thinned.append(float(values[-1]))
    return thinned


def summarise(history: dict) -> dict:
    return {
        "params": history["params"],
        "init_loss": history["init_loss"],
        "init_accuracy": history["init_accuracy"],
        "final_loss": history["final_loss"],
        "final_accuracy": history["final_accuracy"],
        "peak_attn_logit": history["peak_attn_logit"],
        "final_attn_logit": history["final_attn_logit"],
        "mean_update_rms": history["mean_update_rms"],
        "wall_seconds": history["wall_seconds"],
        "optimizer_seconds": history["optimizer_seconds"],
        "ms_per_step": 1e3 * history["wall_seconds"] / len(history["losses"]),
        "optimizer_ms_per_step": 1e3 * history["optimizer_seconds"] / len(history["losses"]),
        "finite": history["finite"],
        "eval_steps": history["eval_steps"],
        "eval_loss": history["eval_loss"],
        "eval_accuracy": history["eval_accuracy"],
        "loss_curve": thin(history["losses"]),
        "max_attn_logit_curve": thin(history["max_attn_logits"]),
        "update_rms_curve": thin(history["update_rms"]),
    }


def accuracy_at(summary: dict, step: int):
    for index, value in enumerate(summary["eval_steps"]):
        if value == step:
            return summary["eval_accuracy"][index]
    return None


def part_lr_sweep(args, task, log: Log) -> dict:
    log("")
    log(f"[tune] {len(ARMS)} arms x {len(LR_GRID['adam'])} learning rates, seed 0, "
        f"{args.steps} steps, chosen on the same budget as the final runs")
    sweep: dict[str, list[dict]] = {}
    tuned: dict[str, float] = {}
    for arm in ARMS:
        sweep[arm] = []
        for learning_rate in LR_GRID[arm]:
            history = run_arm(args, task, arm, learning_rate, seed=0, steps=args.steps, eval_every=0)
            sweep[arm].append(
                {
                    "learning_rate": learning_rate,
                    "final_loss": history["final_loss"],
                    "final_accuracy": history["final_accuracy"],
                    "peak_attn_logit": history["peak_attn_logit"],
                }
            )
            log(
                f"  {arm:13s} lr {learning_rate:<8g} loss {history['final_loss']:8.4f}  "
                f"acc {history['final_accuracy']:.4f}  peak_logit {history['peak_attn_logit']:10.2f}"
            )
        # highest accuracy, then lowest loss, then the smallest learning rate
        best = sorted(
            sweep[arm],
            key=lambda row: (-row["final_accuracy"], row["final_loss"], row["learning_rate"]),
        )[0]
        tuned[arm] = best["learning_rate"]
        log(f"  -> {arm:13s} tuned learning rate {tuned[arm]:g}")
    return {"lr_sweep": sweep, "tuned_learning_rates": tuned}


def part_main(args, task, log: Log, tuned: dict) -> dict:
    log("")
    log(f"[main] {len(args.seeds)} seeds x {len(ARMS)} arms x {args.steps} steps, "
        f"identical data order and initialisation per seed, eval every {args.eval_every} steps")
    runs: dict[str, dict] = {}
    for seed in args.seeds:
        runs[str(seed)] = {}
        for arm in ARMS:
            history = run_arm(
                args,
                task,
                arm,
                tuned[arm],
                seed=seed,
                steps=args.steps,
                eval_every=args.eval_every,
            )
            summary = summarise(history)
            runs[str(seed)][arm] = summary
            log(
                f"  seed {seed} {arm:13s} lr {tuned[arm]:<8g} loss {summary['final_loss']:8.4f}  "
                f"acc {summary['final_accuracy']:.4f}  acc@100 "
                f"{accuracy_at(summary, 100) if accuracy_at(summary, 100) is not None else float('nan'):.4f}  "
                f"peak_logit {summary['peak_attn_logit']:10.2f}  "
                f"{summary['ms_per_step']:7.2f} ms/step "
                f"(opt {summary['optimizer_ms_per_step']:.2f})"
            )

    table = {}
    for arm in ARMS:
        summaries = [runs[str(seed)][arm] for seed in args.seeds]
        acc100 = [accuracy_at(s, 100) for s in summaries]
        table[arm] = {
            "learning_rate": tuned[arm],
            "final_loss": mean_std([s["final_loss"] for s in summaries]),
            "final_accuracy": mean_std([s["final_accuracy"] for s in summaries]),
            "accuracy_at_100": mean_std([a for a in acc100 if a is not None]),
            "peak_attn_logit": mean_std([s["peak_attn_logit"] for s in summaries]),
            "final_attn_logit": mean_std([s["final_attn_logit"] for s in summaries]),
            "mean_update_rms": mean_std([s["mean_update_rms"] for s in summaries]),
            "ms_per_step": mean_std([s["ms_per_step"] for s in summaries]),
            "optimizer_ms_per_step": mean_std([s["optimizer_ms_per_step"] for s in summaries]),
            "wall_seconds": mean_std([s["wall_seconds"] for s in summaries]),
        }
    return {"runs": runs, "table": table}


def part_scaling(args, log: Log) -> dict:
    log("")
    log(f"[scaling] one optimizer step on a single square matrix, sizes {SCALING_SIZES}")
    started = time.perf_counter()
    sweep = scaling_sweep(SCALING_ARMS, args.scaling_sizes, reps=args.scaling_reps, batches=3, progress=log)
    for arm, entry in sweep["arms"].items():
        top = loglog_slope(args.scaling_sizes[-3:], entry["seconds_per_step"][-3:])
        entry["slope_top"] = top
        log(
            f"  {arm:17s} slope(all) {entry['slope']:.2f}  slope(last 3 sizes) {top:.2f}  "
            f"macs@1024 {analytic_macs(arm, 1024):.3g}"
        )
    sweep["seconds"] = time.perf_counter() - started
    log(f"  sweep took {sweep['seconds']:.1f} s")
    return sweep


def part_spike(args, log: Log) -> dict:
    """A controlled heavy tail: one spike of growing size in a dense gradient."""
    log("")
    log("[brake] ||update||_2 for a dense 64x64 gradient with one entry spiked by s")
    mx.random.seed(0)
    base = mx.random.normal(shape=(64, 64))
    weight = mx.zeros((64, 64))
    rows = []
    arms = {
        "auon_temp": lambda: AuON(learning_rate=1.0, momentum=0.0, **TEMPERATURE_SCALED),
        "auon_vanilla": lambda: AuON(learning_rate=1.0, momentum=0.0, **VANILLA),
        "hybrid_auon": lambda: HybridAuON(
            learning_rate=1.0, momentum=0.0, ns_steps=5, **TEMPERATURE_SCALED
        ),
        "muon": lambda: Muon(learning_rate=1.0, momentum=0.0, weight_decay=0.0),
    }
    for spike in args.spike_sizes:
        gradient = base.at[0, 0].add(spike)
        row = {"spike": spike, "arms": {}}
        raw = {}
        for name, factory in arms.items():
            optimizer = factory()
            statistic = None
            if hasattr(optimizer, "rescale"):
                _, r = optimizer.rescale(gradient)
                mx.eval(r)
                statistic = float(r)
            update = optimizer.apply_gradients({"w": gradient}, {"w": weight})["w"]
            mx.eval(update)
            delta = weight - update
            raw[name] = statistic
            # cosh overflows to inf for a large enough spike, which sends the
            # update to exactly zero; keep the JSON strictly valid (no Infinity)
            row["arms"][name] = {
                "brake_statistic": (
                    statistic
                    if statistic is not None and math.isfinite(statistic)
                    else None
                ),
                "brake_statistic_infinite": bool(
                    statistic is not None and not math.isfinite(statistic)
                ),
                "update_spectral_norm": float(spectral_norm(delta, iters=300)),
                "finite": bool(mx.all(mx.isfinite(delta))),
            }
        rows.append(row)
        log(
            "  spike {:>9g} | ".format(spike)
            + " | ".join(
                f"{name}: "
                + (f"r {raw[name]:.4g} " if raw[name] is not None else "r n/a ")
                + f"||U||2 {row['arms'][name]['update_spectral_norm']:.4g}"
                for name in arms
            )
        )
    return {"rows": rows}


def part_brake(args, task, log: Log, tuned: dict) -> dict:
    log("")
    log(f"[stress] {args.brake_multiplier}x the tuned learning rate for {args.brake_steps} steps, seed 0")
    rows = {}
    for arm in ARMS:
        learning_rate = tuned[arm] * args.brake_multiplier
        history = run_arm(
            args, task, arm, learning_rate, seed=0, steps=args.brake_steps, eval_every=0
        )
        rows[arm] = {
            "learning_rate": learning_rate,
            "final_loss": history["final_loss"],
            "final_accuracy": history["final_accuracy"],
            "peak_attn_logit": history["peak_attn_logit"],
            "final_attn_logit": history["final_attn_logit"],
            "mean_update_rms": history["mean_update_rms"],
            "finite": history["finite"],
        }
        log(
            f"  {arm:13s} lr {learning_rate:<8g} loss {history['final_loss']:10.4f}  "
            f"acc {history['final_accuracy']:.4f}  peak_logit {history['peak_attn_logit']:12.2f}  "
            f"upd_rms {history['mean_update_rms']:.3e}  finite {history['finite']}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-sequences", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--repeat-len", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--adam-learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--scaling-sizes", type=int, nargs="+", default=SCALING_SIZES)
    parser.add_argument("--scaling-reps", type=int, default=3)
    parser.add_argument("--spike-sizes", type=float, nargs="+", default=[0.0, 1.0, 10.0, 100.0, 1e6])
    parser.add_argument("--brake-multiplier", type=float, default=10.0)
    parser.add_argument("--brake-steps", type=int, default=200)
    parser.add_argument("--parts", nargs="+", default=["tune", "main", "scaling", "spike", "brake"])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--quick", action="store_true", help="tiny settings, for a smoke run")
    args = parser.parse_args()

    if args.quick:
        args.steps = 20
        args.seeds = [0]
        args.eval_every = 10
        args.eval_sequences = 32
        args.batch_size = 8
        args.scaling_sizes = [64, 128]
        args.scaling_reps = 1
        args.brake_steps = 10
        args.spike_sizes = [0.0, 10.0]

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    args.out.mkdir(parents=True, exist_ok=True)
    log = Log(args.out / "experiment_log.txt")
    started = time.perf_counter()

    task = build_task(args)
    model = make_model(0, vocab_size=task.n_tokens, **model_kwargs(args))
    log("AuON vs orthogonal momentum baselines on a toy induction task")
    log(f"  mlx device      {mx.default_device()}")
    log(f"  task            induction/copy, vocab {task.n_tokens}, repeat {args.repeat_len}, "
        f"sequence {task.seq_len}, {task.repeat_len} scored positions per sequence")
    log(f"  model           {args.n_layers} layers, d_model {args.d_model}, {args.n_heads} heads, "
        f"d_ff {args.d_ff}, {model.num_params()} parameters")
    log(f"  budget          {args.steps} steps x batch {args.batch_size}, "
        f"{len(args.seeds)} seeds {args.seeds}, eval on {args.eval_sequences} fixed sequences")
    log(f"  arms            {', '.join(ARMS)}")
    log(f"  learning rate   tuned per arm (see the sweep below); Adam on embeddings, "
        f"the LM head and 1-D parameters for every non-Adam arm, lr {args.adam_learning_rate}")
    log(f"  AuON config     rho {args.rho}, vanilla {VANILLA}, temperature-scaled {TEMPERATURE_SCALED}")
    log(f"  Hybrid-AuON     temperature-scaled with {args.ns_steps} Newton-Schulz iterations on 2-D only")

    results: dict = {
        "meta": {
            "device": str(mx.default_device()),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
            "params": model.num_params(),
            "sequence_length": task.seq_len,
            "scored_positions": task.repeat_len,
            "arms": ARMS,
            "adam_learning_rate": args.adam_learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "rho": args.rho,
            "ns_steps": args.ns_steps,
            "vanilla_config": VANILLA,
            "temperature_scaled_config": TEMPERATURE_SCALED,
            "lr_grid": LR_GRID,
        }
    }

    tuned = {arm: LR_GRID[arm][len(LR_GRID[arm]) // 2] for arm in ARMS}
    if "tune" in args.parts:
        results.update(part_lr_sweep(args, task, log))
        tuned = results["tuned_learning_rates"]
    if "main" in args.parts:
        results.update(part_main(args, task, log, tuned))
    if "scaling" in args.parts:
        results["scaling"] = part_scaling(args, log)
    if "spike" in args.parts:
        results["spike"] = part_spike(args, log)
    if "brake" in args.parts:
        results["brake"] = part_brake(args, task, log, tuned)

    results["meta"]["seconds"] = time.perf_counter() - started
    log("")
    log(f"total wall clock {results['meta']['seconds']:.1f} s")
    (args.out / "experiment.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    log.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
