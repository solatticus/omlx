# SPDX-License-Identifier: Apache-2.0
"""
TurboQuant KV cache compression.

Implements the MSE-optimal quantizer from "TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate" (Zandieh et al., ICLR 2026).

Algorithm:
1. Multiply vector by random rotation matrix (QR of random normal)
2. Each rotated coordinate follows Beta ≈ N(0, 1/d) in high dimensions
3. Quantize each coordinate independently using Lloyd-Max centroids
4. Store b-bit indices + vector norm (for rescaling)
5. Dequantize: look up centroids, rotate back, rescale

The rotation matrix and codebook are computed once per (dimension, bits)
pair and reused for all tokens/heads/layers.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# Well-known Lloyd-Max centroids for N(0,1) at common bit-widths.
# These are the optimal scalar quantizer centroids for a standard
# normal distribution. We scale by 1/sqrt(d) for the actual codebook.
_GAUSSIAN_LLOYD_MAX = {
    1: [-0.7979, 0.7979],
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1520, -1.3440, -0.7560, -0.2451,
         0.2451,  0.7560,  1.3440,  2.1520],
    4: [-2.7326, -2.0690, -1.6180, -1.2562,
        -0.9424, -0.6568, -0.3881, -0.1284,
         0.1284,  0.3881,  0.6568,  0.9424,
         1.2562,  1.6180,  2.0690,  2.7326],
}

# Corresponding decision boundaries (midpoints between centroids)
# Used for fast quantization: find which interval a value falls in
_GAUSSIAN_LLOYD_MAX_BOUNDARIES = {}
for _b, _c in _GAUSSIAN_LLOYD_MAX.items():
    _bounds = [(_c[i] + _c[i + 1]) / 2.0 for i in range(len(_c) - 1)]
    _GAUSSIAN_LLOYD_MAX_BOUNDARIES[_b] = _bounds


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression."""
    bits: int = 3
    seed: int = 42


class TurboQuant:
    """
    KV cache compressor using TurboQuant algorithm.

    Compresses KV tensors via random rotation + optimal scalar quantization.
    Near-zero quality loss at 3-4 bits per coordinate with 4-5x compression.
    """

    def __init__(self, config: Optional[TurboQuantConfig] = None):
        self.config = config or TurboQuantConfig()
        self._rotations: Dict[int, mx.array] = {}
        self._codebooks: Dict[Tuple[int, int], mx.array] = {}
        self._boundaries: Dict[Tuple[int, int], mx.array] = {}

    def _get_rotation(self, d: int) -> mx.array:
        """Get or create orthogonal rotation matrix for dimension d."""
        if d not in self._rotations:
            # Deterministic seed per dimension
            key = mx.random.key(self.config.seed + d)
            random_mat = mx.random.normal(shape=(d, d), key=key)
            q, _ = mx.linalg.qr(random_mat, stream=mx.cpu)
            mx.eval(q)
            self._rotations[d] = q
            logger.debug(f"TurboQuant: created {d}x{d} rotation matrix")
        return self._rotations[d]

    def _get_codebook(self, d: int) -> Tuple[mx.array, mx.array]:
        """Get centroids and decision boundaries for (bits, d)."""
        bits = self.config.bits
        key = (bits, d)
        if key not in self._codebooks:
            if bits not in _GAUSSIAN_LLOYD_MAX:
                raise ValueError(f"No precomputed centroids for {bits} bits")

            scale = 1.0 / math.sqrt(d)
            centroids = mx.array(
                [c * scale for c in _GAUSSIAN_LLOYD_MAX[bits]],
                dtype=mx.float32,
            )
            boundaries = mx.array(
                [b * scale for b in _GAUSSIAN_LLOYD_MAX_BOUNDARIES[bits]],
                dtype=mx.float32,
            )
            mx.eval(centroids, boundaries)
            self._codebooks[key] = centroids
            self._boundaries[key] = boundaries
            logger.debug(
                f"TurboQuant: created codebook for {bits}-bit, d={d} "
                f"({len(_GAUSSIAN_LLOYD_MAX[bits])} levels, "
                f"scale={scale:.6f})"
            )
        return self._codebooks[key], self._boundaries[key]

    def compress_tensor(self, x: mx.array) -> Dict[str, mx.array]:
        """
        Compress a single KV tensor.

        Args:
            x: tensor of shape (..., d) where d is head_dim

        Returns:
            dict with 'indices' (uint8), 'norms' (float16), 'shape' (original shape)
        """
        d = x.shape[-1]
        orig_shape = x.shape
        orig_dtype = x.dtype
        # Flatten to (N, d) for batch processing
        flat = x.reshape(-1, d).astype(mx.float32)

        # Compute and store norms for rescaling
        norms = mx.sqrt((flat * flat).sum(axis=-1, keepdims=True))
        # Avoid division by zero
        safe_norms = mx.where(norms > 1e-10, norms, mx.ones_like(norms))
        normalized = flat / safe_norms

        # Rotate: y = Π · x (each row is a vector)
        rotation = self._get_rotation(d)
        rotated = mx.matmul(normalized, rotation.T)

        # Quantize each coordinate to nearest centroid
        codebook, boundaries = self._get_codebook(d)
        n_levels = len(_GAUSSIAN_LLOYD_MAX[self.config.bits])

        # Vectorized bucket assignment using boundaries
        # For each value, count how many boundaries it exceeds
        # boundaries shape: (n_levels-1,), rotated shape: (N, d)
        # Compare rotated[..., None] > boundaries[None, None, :]
        expanded = rotated[:, :, None]  # (N, d, 1)
        bounds = boundaries[None, None, :]  # (1, 1, n_levels-1)
        indices = (expanded > bounds).sum(axis=-1).astype(mx.uint8)  # (N, d)

        return {
            "indices": indices,
            "norms": norms.squeeze(-1).astype(mx.float16),
            "shape": mx.array(list(orig_shape), dtype=mx.int32),
            "dtype": str(orig_dtype),
        }

    def decompress_tensor(self, compressed: Dict[str, mx.array]) -> mx.array:
        """
        Decompress a tensor from quantized representation.

        Args:
            compressed: dict from compress_tensor()

        Returns:
            Approximate reconstruction of original tensor
        """
        indices = compressed["indices"]  # (N, d)
        norms = compressed["norms"]  # (N,)
        orig_shape = tuple(compressed["shape"].tolist())
        d = indices.shape[-1]

        # Look up centroids
        codebook, _ = self._get_codebook(d)
        # indices are uint8 [0, n_levels-1], use as array indices
        dequantized = codebook[indices.astype(mx.int32)]  # (N, d)

        # Rotate back: x̃ = Π^T · ỹ
        rotation = self._get_rotation(d)
        reconstructed = mx.matmul(dequantized, rotation)  # Π^T = Π (orthogonal)

        # Rescale by original norms
        reconstructed = reconstructed * norms[:, None]

        # Cast back to original dtype (e.g., bfloat16) to avoid
        # 2x memory bloat and slower attention during generation
        orig_dtype_str = compressed.get("dtype", "mlx.core.float32")
        if "bfloat16" in orig_dtype_str:
            reconstructed = reconstructed.astype(mx.bfloat16)
        elif "float16" in orig_dtype_str:
            reconstructed = reconstructed.astype(mx.float16)

        return reconstructed.reshape(orig_shape)

    def compress_layer(
        self, keys: mx.array, values: mx.array
    ) -> Dict[str, Any]:
        """
        Compress a layer's KV tensors.

        Handles any shape as long as last dim is head_dim.
        Returns dict preserving metadata for reconstruction.
        """
        return {
            "keys": self.compress_tensor(keys),
            "values": self.compress_tensor(values),
            "compressed": True,
            "bits": self.config.bits,
        }

    def decompress_layer(
        self, compressed: Dict[str, Any]
    ) -> Tuple[mx.array, mx.array]:
        """Decompress a layer's KV tensors."""
        keys = self.decompress_tensor(compressed["keys"])
        values = self.decompress_tensor(compressed["values"])
        return keys, values


def compress_extracted_cache(
    extracted: List[Dict[str, Any]], tq: TurboQuant
) -> List[Dict[str, Any]]:
    """
    Compress an entire extracted cache (all layers).

    Replaces state tensors with compressed representations.
    Preserves meta_state, class_name, cache_type unchanged.
    """
    compressed = []
    for layer in extracted:
        state = layer.get("state", ())
        class_name = layer.get("class_name", "KVCache")

        if class_name == "CacheList":
            # CacheList: state is list of sub-states, each is (keys, values)
            if isinstance(state, (list, tuple)):
                compressed_subs = []
                for sub_state in state:
                    if (
                        isinstance(sub_state, (list, tuple))
                        and len(sub_state) >= 2
                        and hasattr(sub_state[0], "shape")
                    ):
                        compressed_subs.append(
                            tq.compress_layer(sub_state[0], sub_state[1])
                        )
                    else:
                        compressed_subs.append(sub_state)
                compressed.append({
                    **layer,
                    "state": compressed_subs,
                    "_tq_compressed": True,
                })
            else:
                compressed.append(layer)
        elif isinstance(state, (list, tuple)) and len(state) >= 2:
            # Standard KVCache: state is (keys, values)
            keys, values = state[0], state[1]
            if hasattr(keys, "shape") and hasattr(values, "shape"):
                comp = tq.compress_layer(keys, values)
                compressed.append({
                    **layer,
                    "state": comp,
                    "_tq_compressed": True,
                })
            else:
                compressed.append(layer)
        else:
            compressed.append(layer)

    return compressed


def decompress_extracted_cache(
    compressed: List[Dict[str, Any]], tq: TurboQuant
) -> List[Dict[str, Any]]:
    """
    Decompress an entire extracted cache back to original format.

    Restores state tensors from compressed representations.
    """
    decompressed = []
    for layer in compressed:
        if not layer.get("_tq_compressed", False):
            decompressed.append(layer)
            continue

        class_name = layer.get("class_name", "KVCache")
        state = layer["state"]

        if class_name == "CacheList":
            restored_subs = []
            for sub_state in state:
                if isinstance(sub_state, dict) and sub_state.get("compressed"):
                    k, v = tq.decompress_layer(sub_state)
                    restored_subs.append((k, v))
                else:
                    restored_subs.append(sub_state)
            new_layer = {
                k: v for k, v in layer.items() if k != "_tq_compressed"
            }
            new_layer["state"] = restored_subs
            decompressed.append(new_layer)
        elif isinstance(state, dict) and state.get("compressed"):
            k, v = tq.decompress_layer(state)
            new_layer = {
                k_: v_ for k_, v_ in layer.items() if k_ != "_tq_compressed"
            }
            new_layer["state"] = (k, v)
            decompressed.append(new_layer)
        else:
            decompressed.append(layer)

    return decompressed
