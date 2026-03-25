# SPDX-License-Identifier: Apache-2.0
"""
Stateful session layer for oMLX.

Sessions hold KV cache state in memory between chat turns, enabling
append-only inference: only new tokens are computed each turn instead
of reprocessing the full conversation history.

Park/resume serializes session KV directly to SSD as safetensors files,
bypassing the paged block cache (which requires full-block alignment).
"""

import enum
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers for meta_state JSON serialization
# ---------------------------------------------------------------------------

def _meta_to_json(meta):
    """Convert meta_state tuple to JSON-safe structure."""
    if isinstance(meta, (list, tuple)):
        return [_meta_to_json(x) for x in meta]
    if isinstance(meta, (int, float, str, bool, type(None))):
        return meta
    return str(meta)


def _json_to_meta(data):
    """Convert JSON structure back to meta_state tuple."""
    if isinstance(data, list):
        return tuple(_json_to_meta(x) for x in data)
    return data


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class SessionState(str, enum.Enum):
    """Lifecycle state of a session."""
    ACTIVE = "active"
    PARKED = "parked"
    PARKING = "parking"
    RESUMING = "resuming"


# ---------------------------------------------------------------------------
# Session manifest — metadata, no KV data
# ---------------------------------------------------------------------------

@dataclass
class SessionManifest:
    """Persistent metadata for a session."""

    session_id: str
    model_name: str
    created_at: float
    last_active: float
    turn_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    token_ids: List[int] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    state: SessionState = SessionState.ACTIVE
    # Legacy field — kept for manifest compat, no longer used by direct SSD path
    parked_block_table_id: Optional[str] = None
    # TTL in seconds — default 6 hours
    ttl: float = 6 * 3600

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionManifest":
        d = dict(d)  # shallow copy
        d["state"] = SessionState(d.get("state", "active"))
        return cls(**d)


# ---------------------------------------------------------------------------
# SessionKVStore — in-memory extracted cache
# ---------------------------------------------------------------------------

class SessionKVStore:
    """
    Holds extracted KV cache tensors in memory, keyed by session ID.

    The extracted_cache format matches Scheduler._extract_cache_states() output:
    a list[dict] where each dict has:
        {'state': (keys, values), 'meta_state': ..., 'class_name': ..., 'cache_type': ...}

    This is separate from the paged SSD hot_cache (which operates at block
    granularity with write-back semantics). Sessions hold the full extracted
    cache as a single unit for instant injection.
    """

    def __init__(self, compressor: Any = None) -> None:
        self._store: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._model_cache_configs: Dict[str, Any] = {}
        self._sizes: Dict[str, int] = {}
        self._total_bytes: int = 0
        self._lock = threading.Lock()
        self._compressor = compressor  # TurboQuant instance or None

    def put(
        self,
        session_id: str,
        extracted_cache: List[Dict[str, Any]],
        model_cache_config: Any = None,
    ) -> None:
        """Store extracted cache for a session. Replaces any existing entry."""
        # Compress KV tensors if compressor is configured
        if self._compressor is not None:
            try:
                from .cache.turboquant import compress_extracted_cache
                extracted_cache = compress_extracted_cache(
                    extracted_cache, self._compressor
                )
            except Exception as e:
                logger.warning(f"KV compression failed, storing uncompressed: {e}")
        size = self._estimate_size(extracted_cache)
        with self._lock:
            # Remove old entry if updating
            if session_id in self._store:
                self._total_bytes -= self._sizes.get(session_id, 0)
            self._store[session_id] = extracted_cache
            self._store.move_to_end(session_id)  # MRU
            self._sizes[session_id] = size
            self._total_bytes += size
            if model_cache_config is not None:
                self._model_cache_configs[session_id] = model_cache_config

    def get(self, session_id: str) -> Tuple[Optional[List[Dict[str, Any]]], Any]:
        """
        Get extracted cache for a session. Non-destructive.
        Returns (extracted_cache, model_cache_config) or (None, None).
        """
        with self._lock:
            if session_id in self._store:
                self._store.move_to_end(session_id)  # update LRU
                extracted = self._store[session_id]
                mcc = self._model_cache_configs.get(session_id)
                # Decompress KV tensors if they were compressed
                if self._compressor is not None and extracted:
                    if any(layer.get("_tq_compressed") for layer in extracted):
                        try:
                            from .cache.turboquant import decompress_extracted_cache
                            extracted = decompress_extracted_cache(
                                extracted, self._compressor
                            )
                        except Exception as e:
                            logger.warning(f"KV decompression failed: {e}")
                return extracted, mcc
            return None, None

    def remove(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Remove and return extracted cache. Returns None if not found."""
        with self._lock:
            extracted = self._store.pop(session_id, None)
            if extracted is not None:
                self._total_bytes -= self._sizes.pop(session_id, 0)
                self._model_cache_configs.pop(session_id, None)
            return extracted

    def contains(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._store

    def memory_usage(self) -> int:
        """Total estimated bytes across all stored sessions."""
        with self._lock:
            return self._total_bytes

    def session_size(self, session_id: str) -> int:
        """Estimated bytes for a specific session."""
        with self._lock:
            return self._sizes.get(session_id, 0)

    def session_ids(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def lru_session_id(self) -> Optional[str]:
        """Return the least-recently-used session ID, or None."""
        with self._lock:
            if self._store:
                # OrderedDict: first item is LRU
                return next(iter(self._store))
            return None

    @staticmethod
    def _estimate_size(extracted_cache: List[Dict[str, Any]]) -> int:
        """Estimate memory footprint of extracted cache tensors."""
        total = 0
        for layer in extracted_cache:
            state = layer.get("state", ())
            # TurboQuant compressed: state is a dict with indices/norms
            if isinstance(state, dict):
                for v in state.values():
                    if hasattr(v, "nbytes"):
                        total += v.nbytes
                    elif isinstance(v, dict):
                        for vv in v.values():
                            if hasattr(vv, "nbytes"):
                                total += vv.nbytes
            elif isinstance(state, (list, tuple)):
                for item in state:
                    if hasattr(item, "nbytes"):
                        total += item.nbytes
                    elif isinstance(item, dict):
                        # Compressed sub-state
                        for v in item.values():
                            if hasattr(v, "nbytes"):
                                total += v.nbytes
                            elif isinstance(v, dict):
                                for vv in v.values():
                                    if hasattr(vv, "nbytes"):
                                        total += vv.nbytes
                    elif isinstance(item, (list, tuple)):
                        for sub in item:
                            if isinstance(sub, (list, tuple)):
                                for t in sub:
                                    if hasattr(t, "nbytes"):
                                        total += t.nbytes
                            elif hasattr(sub, "nbytes"):
                                total += sub.nbytes
        return total


# ---------------------------------------------------------------------------
# SessionManager — orchestrator
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages session lifecycle: create, chat, park, resume, delete.

    One instance per server, held in ServerState.
    """

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        default_ttl: float = 6 * 3600,
        enable_kv_compression: bool = False,
        kv_compression_bits: int = 3,
    ) -> None:
        self._manifests: Dict[str, SessionManifest] = {}
        compressor = None
        if enable_kv_compression:
            try:
                from .cache.turboquant import TurboQuant, TurboQuantConfig
                compressor = TurboQuant(TurboQuantConfig(bits=kv_compression_bits))
                logger.info(
                    f"Session KV compression enabled: TurboQuant {kv_compression_bits}-bit"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize KV compression: {e}")
        self._kv_store = SessionKVStore(compressor=compressor)
        self._state_dir = state_dir
        self._default_ttl = default_ttl
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        if state_dir:
            state_dir.mkdir(parents=True, exist_ok=True)
            self._load_manifests()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SessionManifest:
        """Create a new session. Returns manifest."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id in self._manifests:
            raise ValueError(f"Session {session_id} already exists")

        now = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        manifest = SessionManifest(
            session_id=session_id,
            model_name=model_name,
            created_at=now,
            last_active=now,
            messages=messages,
            ttl=self._default_ttl,
        )
        self._manifests[session_id] = manifest
        self._get_lock(session_id)  # pre-create lock
        self._persist_manifest(manifest)
        logger.info(f"Session created: {session_id} (model={model_name})")
        return manifest

    def get_session(self, session_id: str) -> Optional[SessionManifest]:
        return self._manifests.get(session_id)

    def list_sessions(self) -> List[SessionManifest]:
        return list(self._manifests.values())

    def delete_session(self, session_id: str) -> bool:
        """Destroy session. Frees memory and removes SSD state."""
        manifest = self._manifests.pop(session_id, None)
        if manifest is None:
            return False

        self._kv_store.remove(session_id)
        self._remove_manifest_file(session_id)
        # Clean up KV directory on SSD
        if self._state_dir:
            kv_dir = self._state_dir / f"{session_id}.kv"
            if kv_dir.exists():
                shutil.rmtree(kv_dir, ignore_errors=True)
        with self._global_lock:
            self._locks.pop(session_id, None)
        logger.info(f"Session deleted: {session_id}")
        return True

    # ------------------------------------------------------------------
    # Cache operations (called from scheduler hooks)
    # ------------------------------------------------------------------

    def prepare_request_cache(
        self,
        session_id: str,
        new_token_ids: List[int],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Any], int, List[int]]:
        """
        Check if session has cached KV and prepare it for injection.

        Returns:
            (extracted_cache, model_cache_config, cached_tokens, remaining_tokens)
            or (None, None, 0, new_token_ids) on miss/first turn.
        """
        manifest = self._manifests.get(session_id)
        if manifest is None or manifest.state != SessionState.ACTIVE:
            return None, None, 0, new_token_ids

        extracted, model_cache_config = self._kv_store.get(session_id)
        if extracted is None:
            # First turn or KV was evicted
            return None, None, 0, new_token_ids

        # Validate prefix match: the session's token_ids should be a
        # prefix of the new request's token_ids (the client re-sends
        # the full conversation each time via chat template).
        cached_len = len(manifest.token_ids)
        if cached_len == 0:
            return None, None, 0, new_token_ids

        if len(new_token_ids) < cached_len:
            logger.warning(
                f"Session {session_id}: new request ({len(new_token_ids)} tokens) "
                f"shorter than cached ({cached_len} tokens) — prefix mismatch"
            )
            return None, None, 0, new_token_ids

        # Find longest common prefix (generation prompt tokens at the end
        # of the stored sequence may not appear in the next turn's template)
        match_len = self._find_divergence(manifest.token_ids, new_token_ids)

        if match_len == 0:
            logger.warning(
                f"Session {session_id}: no token prefix match at all — "
                f"falling back to prefix cache"
            )
            self._kv_store.remove(session_id)
            manifest.token_ids = []
            return None, None, 0, new_token_ids

        if match_len < cached_len:
            logger.debug(
                f"Session {session_id}: partial prefix match "
                f"{match_len}/{cached_len} tokens "
                f"(generation prompt trimmed)"
            )

        remaining = new_token_ids[match_len:]
        return extracted, model_cache_config, match_len, remaining

    def update_after_generation(
        self,
        session_id: str,
        prompt_token_ids: List[int],
        output_token_ids: List[int],
        extracted_cache: List[Dict[str, Any]],
        model_cache_config: Any = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """
        Called by scheduler after generation completes for a session request.
        Stores the extracted KV cache and updates the manifest.
        """
        manifest = self._manifests.get(session_id)
        if manifest is None:
            logger.error(f"update_after_generation: session {session_id} not found")
            return

        # The full token sequence is prompt + output
        # (prompt_token_ids already includes prior turns via chat template)
        manifest.token_ids = list(prompt_token_ids) + list(output_token_ids)
        manifest.turn_count += 1
        manifest.total_prompt_tokens += prompt_tokens
        manifest.total_completion_tokens += completion_tokens
        manifest.last_active = time.time()

        if messages is not None:
            manifest.messages = messages

        # Store extracted cache in memory
        self._kv_store.put(session_id, extracted_cache, model_cache_config)
        self._persist_manifest(manifest)

        logger.debug(
            f"Session {session_id}: turn {manifest.turn_count}, "
            f"{manifest.total_tokens} tokens cached, "
            f"~{self._kv_store.session_size(session_id) / (1024*1024):.1f}MB"
        )

    # ------------------------------------------------------------------
    # Park / Resume — direct safetensors serialization
    # ------------------------------------------------------------------

    def _flatten_extracted_to_tensors(
        self, extracted: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Flatten extracted cache into a flat {key: mx.array} dict for safetensors,
        plus a metadata list describing the structure for reconstruction.
        """
        tensors = {}
        layer_meta = []

        for i, layer in enumerate(extracted):
            class_name = layer.get("class_name", "KVCache")
            cache_type = layer.get("cache_type", class_name)
            meta_state = layer.get("meta_state", ())
            state = layer.get("state", ())

            lm: Dict[str, Any] = {
                "class_name": class_name,
                "cache_type": cache_type,
                "meta_state": _meta_to_json(meta_state),
                "tensor_keys": [],
            }

            if class_name == "CacheList":
                # state is list of sub-states
                # meta_state is (sub_class_names, sub_meta_states)
                sub_class_names = (
                    meta_state[0] if meta_state and len(meta_state) > 0 else []
                )
                sub_meta_states = (
                    meta_state[1] if meta_state and len(meta_state) > 1 else []
                )
                sub_meta_list = []
                for j, sub_state in enumerate(state):
                    sub_cn = (
                        sub_class_names[j]
                        if j < len(sub_class_names)
                        else "KVCache"
                    )
                    sub_ms = (
                        sub_meta_states[j]
                        if j < len(sub_meta_states)
                        else ()
                    )
                    sub_keys = []
                    if isinstance(sub_state, (list, tuple)):
                        for k, tensor in enumerate(sub_state):
                            if hasattr(tensor, "nbytes"):
                                key = f"l{i}.c{j}.t{k}"
                                tensors[key] = tensor
                                sub_keys.append(key)
                    sub_meta_list.append({
                        "class_name": sub_cn,
                        "meta_state": _meta_to_json(sub_ms),
                        "tensor_keys": sub_keys,
                    })
                lm["sub_caches"] = sub_meta_list

            elif isinstance(state, (list, tuple)):
                for k, tensor in enumerate(state):
                    if hasattr(tensor, "nbytes"):
                        key = f"l{i}.t{k}"
                        tensors[key] = tensor
                        lm["tensor_keys"].append(key)

            layer_meta.append(lm)

        return tensors, layer_meta

    def _unflatten_tensors_to_extracted(
        self, tensors: Dict[str, Any], layer_meta: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Reverse of _flatten_extracted_to_tensors.
        Rebuilds the extracted cache format from flat tensors + metadata.
        """
        extracted = []

        for lm in layer_meta:
            class_name = lm["class_name"]
            cache_type = lm.get("cache_type", class_name)
            meta_state = _json_to_meta(lm["meta_state"])

            if class_name == "CacheList":
                sub_states = []
                sub_class_names = []
                sub_meta_states = []
                for sub in lm.get("sub_caches", []):
                    sub_tensors = tuple(
                        tensors[k] for k in sub["tensor_keys"]
                    )
                    sub_states.append(sub_tensors)
                    sub_class_names.append(sub["class_name"])
                    sub_meta_states.append(_json_to_meta(sub["meta_state"]))

                extracted.append({
                    "state": sub_states,
                    "meta_state": (sub_class_names, sub_meta_states),
                    "class_name": "CacheList",
                    "cache_type": "CacheList",
                })
            else:
                state = tuple(tensors[k] for k in lm["tensor_keys"])
                extracted.append({
                    "state": state,
                    "meta_state": meta_state,
                    "class_name": class_name,
                    "cache_type": cache_type,
                })

        return extracted

    def park_session(self, session_id: str, **kwargs) -> bool:
        """
        Park session: serialize KV tensors directly to SSD via safetensors.
        Bypasses the paged block cache entirely — no block-size alignment needed.
        """
        manifest = self._manifests.get(session_id)
        if manifest is None or manifest.state != SessionState.ACTIVE:
            return False

        _, mcc = self._kv_store.get(session_id)
        extracted = self._kv_store.remove(session_id)
        if extracted is None:
            logger.warning(
                f"Park {session_id}: no KV in memory (already parked or first turn)"
            )
            manifest.state = SessionState.PARKED
            self._persist_manifest(manifest)
            return True

        manifest.state = SessionState.PARKING

        # Decompress if TurboQuant compressed — park stores raw tensors
        # (SSD space is cheap, clean round-trip is more important)
        if any(layer.get("_tq_compressed") for layer in extracted):
            try:
                from .cache.turboquant import decompress_extracted_cache
                extracted = decompress_extracted_cache(
                    extracted, self._kv_store._compressor
                )
                logger.debug(f"Park {session_id}: decompressed KV for serialization")
            except Exception as e:
                logger.error(f"Park {session_id}: decompression failed: {e}")
                self._kv_store.put(session_id, extracted, mcc)
                manifest.state = SessionState.ACTIVE
                return False

        if self._state_dir is None:
            logger.error(f"Park {session_id}: no state_dir configured")
            self._kv_store.put(session_id, extracted, mcc)
            manifest.state = SessionState.ACTIVE
            return False

        kv_dir = self._state_dir / f"{session_id}.kv"

        try:
            import mlx.core as mx

            kv_dir.mkdir(parents=True, exist_ok=True)

            # Flatten to tensors + metadata
            tensors, layer_meta = self._flatten_extracted_to_tensors(extracted)

            if tensors:
                mx.save_safetensors(
                    str(kv_dir / "tensors.safetensors"), tensors
                )

            # Save structure metadata
            meta = {
                "num_layers": len(layer_meta),
                "layers": layer_meta,
                "total_tokens": manifest.total_tokens,
                "model_name": manifest.model_name,
            }
            # Atomic write for meta.json
            fd, tmp = tempfile.mkstemp(
                dir=str(kv_dir), suffix=".tmp", prefix=".meta-"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(meta, f, indent=2)
                os.rename(tmp, str(kv_dir / "meta.json"))
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

            manifest.state = SessionState.PARKED
            self._persist_manifest(manifest)

            tensor_bytes = sum(
                t.nbytes for t in tensors.values() if hasattr(t, "nbytes")
            )
            logger.info(
                f"Session {session_id} parked "
                f"({manifest.total_tokens} tokens, "
                f"{tensor_bytes / (1024*1024):.1f}MB -> {kv_dir})"
            )
            return True

        except Exception as e:
            logger.error(f"Park {session_id} failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Put KV back in memory
            self._kv_store.put(session_id, extracted, mcc)
            manifest.state = SessionState.ACTIVE
            return False

    def resume_session(self, session_id: str, **kwargs) -> bool:
        """
        Resume session: load KV tensors from SSD safetensors.
        Bypasses the paged block cache entirely.
        """
        manifest = self._manifests.get(session_id)
        if manifest is None or manifest.state != SessionState.PARKED:
            return False

        if self._state_dir is None:
            logger.error(f"Resume {session_id}: no state_dir configured")
            return False

        kv_dir = self._state_dir / f"{session_id}.kv"
        meta_path = kv_dir / "meta.json"

        if not meta_path.exists():
            logger.error(
                f"Resume {session_id}: no KV data at {kv_dir} "
                f"(session may have been parked without KV)"
            )
            return False

        manifest.state = SessionState.RESUMING

        try:
            import mlx.core as mx

            # Load structure metadata
            with open(meta_path) as f:
                meta = json.load(f)

            # Load tensors
            tensors_path = kv_dir / "tensors.safetensors"
            if tensors_path.exists():
                tensors = mx.load(str(tensors_path))
            else:
                tensors = {}

            # Reconstruct extracted cache format
            extracted = self._unflatten_tensors_to_extracted(
                tensors, meta["layers"]
            )

            if not extracted:
                logger.error(f"Resume {session_id}: reconstructed empty cache")
                manifest.state = SessionState.PARKED
                return False

            # put() will recompress via TurboQuant if compressor is active
            self._kv_store.put(session_id, extracted)
            manifest.state = SessionState.ACTIVE
            manifest.last_active = time.time()
            self._persist_manifest(manifest)

            tensor_bytes = sum(
                t.nbytes for t in tensors.values() if hasattr(t, "nbytes")
            )
            mem_bytes = self._kv_store.session_size(session_id)
            logger.info(
                f"Session {session_id} resumed "
                f"({manifest.total_tokens} tokens, "
                f"{tensor_bytes / (1024*1024):.1f}MB from SSD, "
                f"{mem_bytes / (1024*1024):.1f}MB in memory)"
            )
            return True

        except Exception as e:
            logger.error(f"Resume {session_id} failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            manifest.state = SessionState.PARKED
            return False

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def get_active_memory(self) -> int:
        """Total bytes used by active session KV caches."""
        return self._kv_store.memory_usage()

    def get_session_memory(self, session_id: str) -> int:
        """Bytes used by a specific session."""
        return self._kv_store.session_size(session_id)

    def evict_lru_session(self, store_cache_fn: Any = None) -> Optional[str]:
        """
        Auto-park the least-recently-used active session.
        Called by ProcessMemoryEnforcer under memory pressure.
        Returns the parked session_id or None.
        """
        lru_id = self._kv_store.lru_session_id()
        if lru_id is None:
            return None

        manifest = self._manifests.get(lru_id)
        if manifest is None or manifest.state != SessionState.ACTIVE:
            return None

        # Try direct SSD park first
        success = self.park_session(lru_id)
        if success:
            return lru_id

        # If park fails, just evict the KV without SSD backup
        logger.warning(f"LRU eviction: park failed for {lru_id}, dropping KV")
        self._kv_store.remove(lru_id)
        manifest.state = SessionState.PARKED
        self._persist_manifest(manifest)
        logger.info(f"LRU eviction: dropped KV for session {lru_id}")
        return lru_id

    def park_all_for_model(
        self, model_name: str, store_cache_fn: Any = None
    ) -> int:
        """Park all active sessions for a model. Returns count parked."""
        count = 0
        for manifest in list(self._manifests.values()):
            if (
                manifest.model_name == model_name
                and manifest.state == SessionState.ACTIVE
            ):
                success = self.park_session(manifest.session_id)
                if not success:
                    # Fallback: just evict KV
                    self._kv_store.remove(manifest.session_id)
                    manifest.state = SessionState.PARKED
                    self._persist_manifest(manifest)
                count += 1
        return count

    def cleanup_expired(self) -> int:
        """Delete expired sessions. Returns count deleted."""
        count = 0
        for sid in list(self._manifests.keys()):
            manifest = self._manifests[sid]
            if manifest.is_expired:
                self.delete_session(sid)
                count += 1
        if count:
            logger.info(f"Cleaned up {count} expired session(s)")
        return count

    # ------------------------------------------------------------------
    # Per-session lock
    # ------------------------------------------------------------------

    def _get_lock(self, session_id: str) -> threading.Lock:
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    def acquire_session_lock(self, session_id: str) -> threading.Lock:
        """Acquire per-session lock for sequential turn enforcement."""
        lock = self._get_lock(session_id)
        lock.acquire()
        return lock

    # ------------------------------------------------------------------
    # Manifest persistence (JSON to state_dir)
    # ------------------------------------------------------------------

    def _persist_manifest(self, manifest: SessionManifest) -> None:
        if self._state_dir is None:
            return
        path = self._state_dir / f"{manifest.session_id}.json"
        try:
            # Atomic write: tmp + rename
            fd, tmp = tempfile.mkstemp(
                dir=self._state_dir, suffix=".tmp", prefix=".session-"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(manifest.to_dict(), f, indent=2)
                os.rename(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.debug(f"Failed to persist manifest {manifest.session_id}: {e}")

    def _remove_manifest_file(self, session_id: str) -> None:
        if self._state_dir is None:
            return
        path = self._state_dir / f"{session_id}.json"
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Failed to remove manifest file {session_id}: {e}")

    def _load_manifests(self) -> None:
        """Load manifests from state_dir on startup."""
        if self._state_dir is None:
            return
        count = 0
        for path in self._state_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                manifest = SessionManifest.from_dict(data)
                # Sessions that were ACTIVE at shutdown lost their in-memory KV
                if manifest.state == SessionState.ACTIVE:
                    manifest.state = SessionState.PARKED
                # Skip expired sessions
                if manifest.is_expired:
                    path.unlink(missing_ok=True)
                    # Also clean up KV directory
                    kv_dir = self._state_dir / f"{manifest.session_id}.kv"
                    if kv_dir.exists():
                        shutil.rmtree(kv_dir, ignore_errors=True)
                    continue
                self._manifests[manifest.session_id] = manifest
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load manifest {path.name}: {e}")
        if count:
            logger.info(f"Loaded {count} session manifest(s) from {self._state_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_divergence(a: List[int], b: List[int]) -> int:
        """Find the index where two token lists first diverge."""
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        return min(len(a), len(b))
