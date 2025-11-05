"""The training loop: schedule, instrumentation contract and reproducibility."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from auon.data import InductionTask
from auon.optim import adam_parameters
from auon.train import TrainConfig, evaluate, lr_at, make_model, train_arm


def tiny_task():
    return InductionTask(vocab_size=4, repeat_len=2)


def tiny_model(task, seed: int = 0):
    return make_model(
        seed,
        vocab_size=task.n_tokens,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=8,
    )


def batches(task, steps, batch_size=8, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield task.batch(rng, batch_size)


def test_learning_rate_schedule_warmup_and_decay():
    config = TrainConfig(steps=100, learning_rate=1.0, warmup_steps=10, decay=True)
    assert lr_at(0, config) == pytest.approx(0.1)
    assert lr_at(9, config) == pytest.approx(1.0)
    assert lr_at(10, config) == pytest.approx(1.0)
    assert lr_at(99, config) == pytest.approx(0.11)
    constant = TrainConfig(steps=100, learning_rate=0.5)
    assert lr_at(0, constant) == 0.5 and lr_at(99, constant) == 0.5


def test_history_contract_and_peak_logit():
    task = tiny_task()
    model = tiny_model(task)
    evaluation = task.fixed_eval(16)
    config = TrainConfig(steps=4, batch_size=8, learning_rate=0.01, eval_every=2, log_every=0)
    history = train_arm("adam", model, batches(task, 4), evaluation, config)
    for key in ("losses", "grad_norms", "max_attn_logits", "update_rms", "learning_rates"):
        assert len(history[key]) == 4, key
    assert history["eval_steps"] == [0, 2, 4]
    assert len(history["eval_loss"]) == 3
    assert history["params"] == model.num_params()
    assert history["peak_attn_logit"] == max(history["max_attn_logits"] + [history["final_attn_logit"]])
    assert history["finite"] is True
    assert history["wall_seconds"] > 0
    assert 0 < history["optimizer_seconds"] < history["wall_seconds"]
    assert history["mean_update_rms"] > 0


def test_training_reduces_the_loss_on_a_copy_task():
    """End-to-end: Adam on the induction task for 80 steps must beat chance."""
    task = InductionTask(vocab_size=4, repeat_len=2)
    model = tiny_model(task)
    evaluation = task.fixed_eval(64)
    config = TrainConfig(steps=80, batch_size=16, learning_rate=0.01, eval_every=40, log_every=0)
    history = train_arm("adam", model, batches(task, 80, 16), evaluation, config)
    assert history["init_accuracy"] < 0.5
    assert history["final_loss"] < history["init_loss"] - 0.3
    assert history["eval_accuracy"][-1] > history["eval_accuracy"][0]


@pytest.mark.parametrize(
    "arm", ["adam", "sgd_momentum", "muon", "auon", "auon_temp", "hybrid_auon"]
)
def test_every_arm_trains_a_few_steps_without_diverging(arm):
    task = tiny_task()
    model = tiny_model(task)
    evaluation = task.fixed_eval(16)
    config = TrainConfig(steps=3, batch_size=8, learning_rate=0.02, log_every=0)
    history = train_arm(arm, model, batches(task, 3), evaluation, config)
    assert history["finite"] is True
    assert all(np.isfinite(history["losses"]))
    assert all(np.isfinite(history["grad_norms"]))
    assert all(value > 0 for value in history["update_rms"])


def test_training_is_reproducible_on_the_cpu():
    """Same seed, same batches, same optimizer: bit-identical losses."""

    def run():
        task = tiny_task()
        model = tiny_model(task)
        config = TrainConfig(steps=5, batch_size=8, learning_rate=0.02, log_every=0)
        return train_arm("auon_temp", model, batches(task, 5), task.fixed_eval(16), config)

    first, second = run(), run()
    assert first["losses"] == second["losses"]
    assert first["final_attn_logit"] == second["final_attn_logit"]


def test_evaluate_reports_loss_accuracy_and_logits():
    task = tiny_task()
    model = tiny_model(task)
    metrics = evaluate(model, task.fixed_eval(16))
    assert set(metrics) == {"loss", "accuracy", "max_attn_logit"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0
    assert np.isfinite(metrics["max_attn_logit"])


def test_the_optimizer_split_sends_embeddings_and_the_head_to_adam():
    task = tiny_task()
    model = tiny_model(task)
    routed = set(adam_parameters(model))
    assert "tok_emb.weight" in routed
    assert "pos_emb.weight" in routed
    assert "head.weight" in routed
    assert "blocks.0.attn.qkv.weight" not in routed
    assert "blocks.0.norm1.weight" in routed  # 1-D parameters go to Adam too
