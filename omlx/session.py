# SPDX-License-Identifier: Apache-2.0
"""
Stateful session layer for oMLX.

Sessions hold KV cache state in memory between chat turns, enabling
append-only inference: only new tokens are computed each turn instead
of reprocessing the full conversation history.

Park/resume serializes session KV to SSD and back, reusing the existing
paged SSD cache infrastructure.
"""

import enum
import json
import logging
import os
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
    # request_id alias used when KV is parked to SSD via block_aware_cache
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

    def __init__(self) -> None:
        self._store: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._model_cache_configs: Dict[str, Any] = {}
        self._sizes: Dict[str, int] = {}
        self._total_bytes: int = 0
        self._lock = threading.Lock()

    def put(
        self,
        session_id: str,
        extracted_cache: List[Dict[str, Any]],
        model_cache_config: Any = None,
    ) -> None:
        """Store extracted cache for a session. Replaces any existing entry."""
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
                return (
                    self._store[session_id],
                    self._model_cache_configs.get(session_id),
                )
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
            if isinstance(state, (list, tuple)):
                for item in state:
                    if hasattr(item, "nbytes"):
                        total += item.nbytes
                    elif isinstance(item, (list, tuple)):
                        # CacheList sub-states
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
    ) -> None:
        self._manifests: Dict[str, SessionManifest] = {}
        self._kv_store = SessionKVStore()
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
    # Park / Resume
    # ------------------------------------------------------------------

    def park_session(
        self,
        session_id: str,
        store_cache_fn: Any,
    ) -> bool:
        """
        Park session: move KV from memory to SSD.

        Args:
            store_cache_fn: Callable matching BlockAwarePrefixCache.store_cache()
                signature: (request_id, token_ids, extracted_cache, ...) -> block_table
        """
        manifest = self._manifests.get(session_id)
        if manifest is None or manifest.state != SessionState.ACTIVE:
            return False

        # Grab model_cache_config before remove clears it
        _, mcc = self._kv_store.get(session_id)
        extracted = self._kv_store.remove(session_id)
        if extracted is None:
            logger.warning(f"Park {session_id}: no KV in memory (already parked or first turn)")
            manifest.state = SessionState.PARKED
            self._persist_manifest(manifest)
            return True

        manifest.state = SessionState.PARKING
        park_id = f"session-{session_id}"

        try:
            block_table = store_cache_fn(
                park_id,
                manifest.token_ids,
                extracted,
                model_cache_config=mcc,
            )
            if block_table is not None:
                manifest.parked_block_table_id = park_id
                manifest.state = SessionState.PARKED
                self._persist_manifest(manifest)
                logger.info(
                    f"Session {session_id} parked "
                    f"({manifest.total_tokens} tokens -> SSD)"
                )
                return True
            else:
                logger.error(f"Park {session_id}: store_cache returned None")
                # Put KV back
                self._kv_store.put(session_id, extracted)
                manifest.state = SessionState.ACTIVE
                return False
        except Exception as e:
            logger.error(f"Park {session_id} failed: {e}")
            # Put KV back
            self._kv_store.put(session_id, extracted)
            manifest.state = SessionState.ACTIVE
            return False

    def resume_session(
        self,
        session_id: str,
        fetch_cache_fn: Any,
        reconstruct_cache_fn: Any,
        extract_cache_states_fn: Any,
    ) -> bool:
        """
        Resume session: load KV from SSD back to memory.

        Args:
            fetch_cache_fn: BlockAwarePrefixCache.fetch_cache(request_id, token_ids)
            reconstruct_cache_fn: BlockAwarePrefixCache.reconstruct_cache(block_table)
            extract_cache_states_fn: Scheduler._extract_cache_states(raw_cache)
        """
        manifest = self._manifests.get(session_id)
        if manifest is None or manifest.state != SessionState.PARKED:
            return False

        park_id = manifest.parked_block_table_id
        if park_id is None:
            logger.error(f"Resume {session_id}: no parked_block_table_id")
            return False

        manifest.state = SessionState.RESUMING

        try:
            block_table, remaining = fetch_cache_fn(park_id, manifest.token_ids)
            if block_table is None or block_table.num_tokens == 0:
                logger.error(f"Resume {session_id}: fetch_cache returned no blocks")
                manifest.state = SessionState.PARKED
                return False

            reconstructed = reconstruct_cache_fn(block_table)
            if reconstructed is None:
                logger.error(f"Resume {session_id}: reconstruct_cache failed")
                manifest.state = SessionState.PARKED
                return False

            # Extract states back into session format
            extracted, mcc = extract_cache_states_fn(reconstructed)
            if not extracted:
                logger.error(f"Resume {session_id}: extract_cache_states returned empty")
                manifest.state = SessionState.PARKED
                return False

            self._kv_store.put(session_id, extracted, mcc)
            manifest.state = SessionState.ACTIVE
            manifest.parked_block_table_id = None
            manifest.last_active = time.time()
            self._persist_manifest(manifest)

            logger.info(
                f"Session {session_id} resumed "
                f"({manifest.total_tokens} tokens, "
                f"~{self._kv_store.session_size(session_id) / (1024*1024):.1f}MB)"
            )
            return True

        except Exception as e:
            logger.error(f"Resume {session_id} failed: {e}")
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

        if store_cache_fn is not None:
            success = self.park_session(lru_id, store_cache_fn)
            if success:
                return lru_id
            # If park fails, just evict the KV without SSD backup
            logger.warning(f"LRU eviction: park failed for {lru_id}, dropping KV")

        # Evict without parking (KV lost, but session manifest survives)
        self._kv_store.remove(lru_id)
        manifest.state = SessionState.PARKED
        self._persist_manifest(manifest)
        logger.info(f"LRU eviction: dropped KV for session {lru_id}")
        return lru_id

    def park_all_for_model(self, model_name: str, store_cache_fn: Any = None) -> int:
        """Park all active sessions for a model. Returns count parked."""
        count = 0
        for manifest in list(self._manifests.values()):
            if manifest.model_name == model_name and manifest.state == SessionState.ACTIVE:
                if store_cache_fn:
                    self.park_session(manifest.session_id, store_cache_fn)
                else:
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
