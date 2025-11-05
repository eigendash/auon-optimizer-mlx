"""Training loop, evaluation and the instrumentation the paper's claims need.

Everything the experiment reports is produced here: per-step loss, gradient
norm, the largest causal attention logit seen in the batch, the RMS of the
parameter change (the size of the step actually taken), and a separate timing of
the optimizer update itself so that the wall-clock numbers are not dominated by
the model's forward pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map, tree_reduce

from .data import Batch, next_token_accuracy
from .model import TinyLM, cross_entropy
from .optim import build_optimizer


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 32
    learning_rate: float = 0.01
    adam_learning_rate: float = 1e-3
    momentum: float = 0.95
    weight_decay: float = 0.0
    rho: float = 1.0
    ns_steps: int = 5
    brake_exponent: float | None = None
    eval_every: int = 0
    log_every: int = 50
    warmup_steps: int = 0
    decay: bool = False


def make_model(seed: int, **kwargs) -> TinyLM:
    """Build a :class:`TinyLM` with a reproducible initialisation."""
    mx.random.seed(seed)
    return TinyLM(**kwargs)


def evaluate(model: TinyLM, batch: Batch) -> dict:
    """Loss, next-token accuracy and the batch's largest attention logit."""
    logits = model(batch.tokens)
    loss = cross_entropy(logits, batch.targets, batch.mask)
    mx.eval(loss)
    return {
        "loss": float(loss),
        "accuracy": next_token_accuracy(logits, batch.targets, batch.mask),
        "max_attn_logit": model.max_attention_logit(batch.tokens),
    }


def lr_at(step: int, config: TrainConfig) -> float:
    """Constant, linear warm-up, then optional linear decay to 10%."""
    lr = config.learning_rate
    if config.warmup_steps and step < config.warmup_steps:
        lr = lr * (step + 1) / config.warmup_steps
    if config.decay:
        progress = max(0.0, (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps))
        lr = lr * (1.0 - 0.9 * progress)
    return lr


def train_arm(
    arm: str,
    model: TinyLM,
    train_batches,
    eval_batch: Batch,
    config: TrainConfig,
    progress=None,
) -> dict:
    """Train one optimizer arm and return its logged history.

    ``train_batches`` is an iterable of :class:`~auon.data.Batch`; the same
    iterable object is consumed once, so the caller controls the data order.
    """
    optimizer = build_optimizer(
        arm,
        config.learning_rate,
        adam_learning_rate=config.adam_learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        rho=config.rho,
        ns_steps=config.ns_steps,
        brake_exponent=config.brake_exponent,
    )

    def loss_fn(model: TinyLM, batch: Batch) -> mx.array:
        return cross_entropy(model(batch.tokens), batch.targets, batch.mask)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    n_params = model.num_params()

    history = {
        "arm": arm,
        "params": n_params,
        "losses": [],
        "grad_norms": [],
        "max_attn_logits": [],
        "update_rms": [],
        "eval_steps": [],
        "eval_loss": [],
        "eval_accuracy": [],
        "learning_rates": [],
    }
    start_eval = evaluate(model, eval_batch)
    history["init_loss"] = start_eval["loss"]
    history["init_accuracy"] = start_eval["accuracy"]
    history["eval_steps"].append(0)
    history["eval_loss"].append(start_eval["loss"])
    history["eval_accuracy"].append(start_eval["accuracy"])

    optimizer_seconds = 0.0
    started = time.perf_counter()
    for step in range(config.steps):
        batch = next(train_batches)
        lr = lr_at(step, config)
        optimizer.learning_rate = mx.array(lr)
        loss, grads = loss_and_grad(model, batch)
        # Evaluate the forward/backward outside the timed region, otherwise the
        # lazy graph would move the whole backward pass into the optimizer timer.
        mx.eval(loss, grads)

        before = model.parameters()
        t0 = time.perf_counter()
        optimizer.update(model, grads)
        # Force the update to complete so the timing and the parameter change are
        # both real; without this MLX would defer everything to the eval below.
        mx.eval(model.parameters())
        optimizer_seconds += time.perf_counter() - t0

        delta = tree_map(lambda a, b: ((b - a) ** 2).sum(), before, model.parameters())
        delta = tree_reduce(lambda acc, d: acc + d, delta, 0.0)
        update_rms = float(mx.sqrt(delta / n_params))
        grad_norm = float(mx.sqrt(tree_reduce(lambda acc, g: acc + g.square().sum(), grads, 0.0)))
        max_logit = model.max_attention_logit(batch.tokens)

        history["losses"].append(float(loss))
        history["grad_norms"].append(grad_norm)
        history["max_attn_logits"].append(max_logit)
        history["update_rms"].append(update_rms)
        history["learning_rates"].append(lr)

        if config.eval_every and (step + 1) % config.eval_every == 0:
            metrics = evaluate(model, eval_batch)
            history["eval_steps"].append(step + 1)
            history["eval_loss"].append(metrics["loss"])
            history["eval_accuracy"].append(metrics["accuracy"])
        if config.log_every and (step + 1) % config.log_every == 0:
            message = (
                f"  step {step + 1:4d}  loss {float(loss):.4f}  "
                f"grad {grad_norm:.3f}  upd_rms {update_rms:.2e}  "
                f"max_logit {max_logit:8.3f}  lr {lr:.4g}"
            )
            if progress:
                progress(message)

    history["wall_seconds"] = time.perf_counter() - started
    history["optimizer_seconds"] = optimizer_seconds
    final = evaluate(model, eval_batch)
    history["final_loss"] = final["loss"]
    history["final_accuracy"] = final["accuracy"]
    history["final_attn_logit"] = final["max_attn_logit"]
    history["peak_attn_logit"] = max(history["max_attn_logits"] + [final["max_attn_logit"]])
    history["mean_update_rms"] = float(sum(history["update_rms"]) / len(history["update_rms"]))
    history["finite"] = bool(
        mx.all(mx.isfinite(mx.array(history["losses"] + history["max_attn_logits"])))
    )
    return history
