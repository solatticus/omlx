# SPDX-License-Identifier: Apache-2.0
"""Train Medusa speculation heads on a frozen backbone.

Usage:
    cd ~/src/omlx
    python -m omlx.tools.train_medusa \
        --model /Users/iz/models/Qwen3.5-VL-122B-A10B-4bit-CRACK \
        --iters 1200 --batch 1

Takes ~45 min on M3 Ultra 96GB.  Outputs a small safetensors file
containing only the head weights.
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load

from omlx.speculative import MedusaHeads


def train_medusa(
    model_path: str,
    num_heads: int = 4,
    iters: int = 1200,
    batch_size: int = 1,
    seq_len: int = 2048,
    lr: float = 1e-4,
    output: str = "medusa_heads.safetensors",
):
    print(f"Loading base model: {model_path}")
    model, tokenizer = load(model_path)[:2]

    print(f"Adding {num_heads} Medusa heads (backbone frozen)")
    medusa = MedusaHeads(model, num_heads=num_heads)
    medusa.init_from_lm_head()

    # Freeze backbone, only train heads
    medusa._inner.freeze()
    if medusa.lm_head is not None:
        medusa.lm_head.freeze()

    # Collect trainable params (only medusa heads)
    trainable = []
    for head in medusa.medusa_heads:
        trainable.extend(head.parameters().values())
    n_params = sum(p.size for p in trainable)
    print(f"Trainable parameters: {n_params:,} ({n_params * 2 / 1e6:.1f} MB at fp16)")

    optimizer = nn.optimizers.AdamW(learning_rate=lr)

    # Load training data
    try:
        from datasets import load_dataset
        print("Loading ShareGPT dataset...")
        dataset = load_dataset(
            "Aeala/ShareGPT_Vicuna_unfiltered",
            split="train[:20000]",
        )
        use_hf = True
    except ImportError:
        print("datasets not installed, using random token sequences for training")
        use_hf = False

    def get_batch():
        if use_hf:
            indices = mx.random.randint(0, len(dataset), (batch_size,)).tolist()
            texts = []
            for idx in indices:
                convos = dataset[int(idx)].get("conversations", [])
                text = " ".join(
                    turn.get("value", "") for turn in convos if isinstance(turn, dict)
                )
                texts.append(text[:8000])
            tokens_list = [tokenizer.encode(t)[:seq_len] for t in texts]
            max_len = max(len(t) for t in tokens_list)
            # Pad to same length
            padded = [t + [0] * (max_len - len(t)) for t in tokens_list]
            return mx.array(padded)
        else:
            vocab_size = medusa._lm.args.vocab_size
            return mx.random.randint(0, vocab_size, (batch_size, seq_len))

    def loss_fn(medusa_model, inputs):
        """Medusa-1 loss: each head predicts the next token at its offset."""
        hidden = medusa_model.forward_hidden(inputs)
        backbone_logits, head_logits_list = medusa_model(hidden)

        total_loss = mx.array(0.0)
        n_terms = 0

        for offset, logits in enumerate([backbone_logits] + head_logits_list):
            # Head i predicts token at position t+i+1
            shift = offset + 1
            if shift >= inputs.shape[1]:
                break
            pred = logits[:, :-shift, :]
            target = inputs[:, shift:]
            # Trim to same length
            min_len = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_len, :]
            target = target[:, :min_len]
            total_loss = total_loss + nn.losses.cross_entropy(
                pred.reshape(-1, pred.shape[-1]),
                target.reshape(-1),
                reduction="mean",
            )
            n_terms += 1

        return total_loss / n_terms

    loss_and_grad = nn.value_and_grad(medusa, loss_fn)

    print(f"Training {num_heads} Medusa heads for {iters} iterations...")
    t0 = time.perf_counter()
    smoothed_loss = None

    for step in range(iters):
        inputs = get_batch()
        loss, grads = loss_and_grad(medusa, inputs)
        optimizer.apply_gradients(grads, medusa)
        mx.eval(loss, medusa.parameters())

        loss_val = loss.item()
        smoothed_loss = loss_val if smoothed_loss is None else 0.95 * smoothed_loss + 0.05 * loss_val

        if step % 100 == 0 or step == iters - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (step + 1) * (iters - step - 1)
            print(
                f"  step {step:>5}/{iters}  "
                f"loss={smoothed_loss:.4f}  "
                f"elapsed={elapsed:.0f}s  "
                f"eta={eta:.0f}s"
            )

    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save only head weights
    medusa.save_heads(output)
    print(f"Medusa heads saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Medusa speculation heads on a frozen backbone"
    )
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--heads", type=int, default=4, help="Number of Medusa heads")
    parser.add_argument("--iters", type=int, default=1200, help="Training iterations")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--output", type=str, default="medusa_heads.safetensors",
        help="Output path for trained head weights",
    )
    args = parser.parse_args()

    train_medusa(
        model_path=args.model,
        num_heads=args.heads,
        iters=args.iters,
        batch_size=args.batch,
        seq_len=args.seq_len,
        lr=args.lr,
        output=args.output,
    )
