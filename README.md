# auon-optimizer-mlx

A small, independent MLX implementation of the AuON optimizer, the method in
*AuON: A Linear-time Alternative to Orthogonal Momentum Updates*
(arXiv:2509.24320, submitted 29 September 2025).  The paper asks what makes
orthogonal momentum updates such as Muon work, and proposes replacing the
approximate orthogonalisation with a linear-time rescaling of the momentum by a
hyperbolic-cosine ("emergency brake") statistic, plus a hybrid variant that
keeps Newton-Schulz iterations only for the matrix parameters.

This repository is **not** a reproduction of the paper's results.  The paper
trains a 40M-parameter nanoGPT on SmolLM-Corpus across four L4 GPUs, a 168M
model on FineWebEDU-10B on an H100, and a 19.9M CNN on CIFAR-10; nothing here is
comparable to that.  What is here is a 183k-parameter transformer, a synthetic
induction task and timing sweeps that run in minutes on an Apple-silicon CPU.

Further notes, the measured numbers and the list of places where this
implementation may differ from the paper are added with the final commit.
