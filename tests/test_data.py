"""The induction/copy task and the built-in corpus."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from auon.data import CORPUS, InductionTask, TinyCorpus, next_token_accuracy


def test_induction_sequence_layout_is_bos_a_sep_a():
    task = InductionTask(vocab_size=8, repeat_len=4)
    batch = task.batch(np.random.default_rng(0), batch_size=3)
    tokens = np.array(batch.tokens)
    assert tokens.shape == (3, 10)
    assert (tokens[:, 0] == task.bos).all()
    assert (tokens[:, 5] == task.sep).all()
    assert (tokens[:, 1:5] == tokens[:, 6:10]).all()
    assert tokens[:, 1:5].max() < task.vocab_size


def test_induction_loss_mask_covers_exactly_the_second_copy():
    task = InductionTask(vocab_size=8, repeat_len=4)
    batch = task.batch(np.random.default_rng(1), batch_size=5)
    tokens = np.array(batch.tokens)
    mask = np.array(batch.mask)
    targets = np.array(batch.targets)
    assert mask.sum() == 5 * task.repeat_len
    # targets are always the next token, and the scored targets are the second copy
    assert (targets[:, :-1] == tokens[:, 1:]).all()
    selected = np.where(mask[0] > 0)[0]
    assert list(selected) == [5, 6, 7, 8]
    assert (targets[0, selected] == tokens[0, 6:10]).all()


def test_induction_batches_are_seed_deterministic():
    task = InductionTask(vocab_size=8, repeat_len=4)
    first = task.batch(np.random.default_rng(7), batch_size=4)
    second = task.batch(np.random.default_rng(7), batch_size=4)
    other = task.batch(np.random.default_rng(8), batch_size=4)
    assert np.array_equal(np.array(first.tokens), np.array(second.tokens))
    assert not np.array_equal(np.array(first.tokens), np.array(other.tokens))
    assert np.array_equal(np.array(first.tokens), np.array(task.fixed_eval(4, seed=7).tokens))


def test_next_token_accuracy_is_hand_computable():
    # argmax matches the target at the first masked position and not the second
    logits = mx.array([[[0.0, 5.0], [5.0, 0.0], [0.0, 5.0]]])
    targets = mx.array([[1, 1, 0]])
    mask = mx.array([[1.0, 1.0, 0.0]])
    assert next_token_accuracy(logits, targets, mask) == 0.5


def test_corpus_windows_are_contiguous_and_inside_the_vocabulary():
    corpus = TinyCorpus(seq_len=16, batch_size=4)
    batch = corpus.batch(np.random.default_rng(0))
    tokens = np.array(batch.tokens)
    targets = np.array(batch.targets)
    assert tokens.shape == (4, 16)
    assert tokens.max() < corpus.n_tokens and targets.max() < corpus.n_tokens
    # next-token targets inside the window, plus the character that follows it
    assert np.array_equal(targets[:, :-1], tokens[:, 1:])
    assert batch.mask.sum() == 4 * 16
    for row, target in zip(tokens, targets):
        starts = [p for p in range(len(corpus.data) - 16) if np.array_equal(corpus.data[p : p + 16], row)]
        assert starts, "window is not a contiguous slice of the corpus"
        assert any(corpus.data[p + 16] == target[-1] for p in starts)
    assert len(CORPUS) > 1000


def test_corpus_and_induction_batches_share_the_batch_contract():
    induction = InductionTask(vocab_size=4, repeat_len=2).fixed_eval(2)
    corpus = TinyCorpus(seq_len=8, batch_size=2).fixed_eval(2)
    for batch in (induction, corpus):
        assert batch.tokens.shape == batch.targets.shape == batch.mask.shape
        assert batch.tokens.dtype == batch.targets.dtype == mx.int32
        assert batch.mask.dtype == mx.float32
