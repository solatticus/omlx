# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding via Apple's Recurrent Drafter (ReDrafter).

Tree-based speculative decoding: the drafter (tiny RNN) proposes a tree of
candidate tokens, the target LLM verifies the entire tree in one forward pass,
and we accept the longest matching branch.  Gives 1.5-2.3x decode throughput
on Apple Silicon with zero quality regression.

Integration notes:
  - Speculation runs in fp16 (tree masks require standard SDPA, not TurboQuant
    fused kernel).  TurboQuant is used for session persistence only.
  - The drafter is conditioned on the target model's hidden states.
  - Requires a pre-trained drafter checkpoint matched to the target model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .recurrent_drafter import (
    attention,
    kv_cache,
    modeling_drafter,
    recurrent_drafting,
    tree_attention,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model adapter: bridge mlx_lm/mlx_vlm model to ReDrafter's interface
# ---------------------------------------------------------------------------

class LLMAdapter(nn.Module):
    """Wraps an mlx_lm or mlx_vlm model so ReDrafter can call it.

    ReDrafter expects:  llm(input_ids, beam_len, mask, cache) -> (hidden_states, logits)
    mlx_lm models:      model(input_ids, cache) -> logits
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        # Unwrap VLM -> LanguageModel -> inner model
        if hasattr(model, "language_model"):
            self._lm = model.language_model            # VLM wrapper
        else:
            self._lm = model                           # plain LLM

        # Inner transformer (returns hidden states) and lm_head (logits)
        self._inner = self._lm.model
        if hasattr(self._lm, "lm_head"):
            self._lm_head = self._lm.lm_head
            self._tied = False
        else:
            self._lm_head = None
            self._tied = True

        # Expose args for ReDrafter (hidden_size, vocab_size, num_hidden_layers, etc.)
        self.args = self._lm.args

    def __call__(
        self,
        input_ids: mx.array,
        beam_len: int = 1,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass returning (hidden_states, logits)."""
        # ReDrafter passes its own kv_cache.Cache.sliced (list of (View, View) tuples).
        # mlx_lm expects a list of KVCache objects.  We bridge via _CacheShim.
        if cache is not None and len(cache) > 0:
            if isinstance(cache[0], tuple):
                cache = [_CacheShim(k_view, v_view) for k_view, v_view in cache]

        h = self._inner(input_ids, cache=cache)

        if self._tied:
            logits = self._inner.embed_tokens.as_linear(h)
        else:
            logits = self._lm_head(h)

        return h, logits

    @property
    def input_embeddings(self):
        return self._inner.embed_tokens

    @property
    def model(self):
        """Alias so ReDrafter can access llm.model.input_embeddings."""
        return self


# ---------------------------------------------------------------------------
# KV cache shim: bridge ReDrafter's View to mlx_lm's KVCache interface
# ---------------------------------------------------------------------------

class _CacheShim:
    """Makes a (kv_cache.View, kv_cache.View) pair look like an mlx_lm KVCache.

    mlx_lm's attention layers call:
      keys, values = cache.update_and_fetch(keys, values)
    ReDrafter's Views expose:
      full_keys = view.cat(new_keys)      # append + return full
    """

    def __init__(self, k_view: kv_cache.View, v_view: kv_cache.View):
        self._k = k_view
        self._v = v_view

    @property
    def offset(self):
        return self._k.length

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append new KV and return full cached KV."""
        full_k = self._k.cat(keys)
        full_v = self._v.cat(values)
        return full_k, full_v

    @property
    def state(self):
        return (
            self._k._cache[:, :, :self._k.length, :],
            self._v._cache[:, :, :self._v.length, :],
        )


# ---------------------------------------------------------------------------
# Speculative generation
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    enabled: bool = False
    drafter_path: Optional[str] = None
    beam_width: int = 3        # number of beams (2-4 optimal on M-series)
    beam_length: int = 5       # tokens per beam (matches drafter training)
    pad_token_id: int = 0
    eos_token_id: int = 151643  # Qwen default


def load_drafter(
    drafter_path: str,
    target_model: nn.Module,
) -> modeling_drafter.Drafter:
    """Load a pre-trained ReDrafter checkpoint.

    Args:
        drafter_path: path to drafter weights dir (contains config.json + safetensors)
        target_model: the target LLM (for hidden_size / vocab_size)
    """
    drafter = modeling_drafter.load_model(drafter_path)
    logger.info(
        f"ReDrafter loaded from {drafter_path} "
        f"(exit_dim={drafter.args.exit_dim}, "
        f"layers={drafter.args.num_draft_layers}, "
        f"rnn={drafter.args.rnn})"
    )
    return drafter


def speculative_generate(
    model: nn.Module,
    drafter: modeling_drafter.Drafter,
    input_ids: mx.array,
    max_tokens: int,
    config: SpeculativeConfig,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Generator[Tuple[mx.array, int], None, None]:
    """Tree-based speculative generation using ReDrafter.

    Wraps the target model with LLMAdapter and delegates to
    ReDrafterModel._generate().

    Args:
        model: the oMLX-loaded model (mlx_lm or mlx_vlm)
        drafter: pre-trained Drafter model
        input_ids: (batch_size, seq_len) prompt tokens
        max_tokens: maximum tokens to generate
        config: speculative decoding configuration
        temperature: sampling temperature (0.0 = greedy)
        top_p: nucleus sampling threshold

    Yields:
        (output_tokens, n_accepted) per speculation step
    """
    adapter = LLMAdapter(model)

    redrafter = recurrent_drafting.ReDrafterModel(
        llm=adapter,
        drafter=drafter,
    )

    beam_shape = modeling_drafter.BeamShape(
        width=config.beam_width,
        length=config.beam_length,
    )
    sampling_args = recurrent_drafting.SamplingArgs(
        temperature=temperature if temperature > 0 else 1.0,
        greedy=(temperature == 0.0),
    )
    special_tokens = recurrent_drafting.SpecialTokens(
        pad=config.pad_token_id,
        eos=config.eos_token_id,
    )

    max_length = input_ids.shape[1] + max_tokens

    prev_len = input_ids.shape[1]
    for output_tokens in redrafter.generate(
        input_ids=input_ids,
        max_length=max_length,
        beam_shape=beam_shape,
        sampling_args=sampling_args,
        special_tokens=special_tokens,
    ):
        n_accepted = output_tokens.shape[1] - prev_len
        prev_len = output_tokens.shape[1]
        if n_accepted > 0:
            yield output_tokens, n_accepted
