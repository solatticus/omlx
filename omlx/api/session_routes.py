# SPDX-License-Identifier: Apache-2.0
"""
Session API routes for stateful KV cache management.

Endpoints:
- POST   /v1/sessions              — Create a new session
- GET    /v1/sessions              — List all sessions
- GET    /v1/sessions/{id}         — Get session details
- POST   /v1/sessions/{id}/chat    — Chat within a session (KV retained)
- POST   /v1/sessions/{id}/park    — Park session KV to SSD
- POST   /v1/sessions/{id}/resume  — Resume session KV from SSD
- DELETE /v1/sessions/{id}         — Delete session
"""

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])

# Dependency injection callbacks (set by server.py)
_get_session_manager = None
_get_engine_pool = None
_get_engine_for_model = None
_get_server_state = None


def set_session_getters(
    session_manager_getter,
    engine_pool_getter,
    engine_for_model_getter,
    server_state_getter,
):
    """Set callback functions for dependency injection."""
    global _get_session_manager, _get_engine_pool
    global _get_engine_for_model, _get_server_state
    _get_session_manager = session_manager_getter
    _get_engine_pool = engine_pool_getter
    _get_engine_for_model = engine_for_model_getter
    _get_server_state = server_state_getter


def _mgr():
    if _get_session_manager is None:
        raise HTTPException(500, "Session manager not initialized")
    mgr = _get_session_manager()
    if mgr is None:
        raise HTTPException(500, "Session manager not initialized")
    return mgr


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None

class CreateSessionResponse(BaseModel):
    session_id: str
    model: str
    created_at: float

class SessionChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None

class SessionChatResponse(BaseModel):
    session_id: str
    content: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = {}
    turn: int = 0
    cached_tokens: int = 0

class SessionInfo(BaseModel):
    session_id: str
    model: str
    state: str
    turn_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    created_at: float
    last_active: float
    memory_bytes: int

class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]
    total_active_memory: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new session."""
    mgr = _mgr()

    # Resolve model name
    model_name = request.model
    if model_name is None:
        state = _get_server_state()
        model_name = state.default_model
    if not model_name:
        raise HTTPException(400, "No model specified and no default model configured")

    try:
        manifest = mgr.create_session(
            model_name=model_name,
            system_prompt=request.system_prompt,
            session_id=request.session_id,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return CreateSessionResponse(
        session_id=manifest.session_id,
        model=manifest.model_name,
        created_at=manifest.created_at,
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions():
    """List all sessions."""
    mgr = _mgr()
    sessions = []
    for m in mgr.list_sessions():
        sessions.append(SessionInfo(
            session_id=m.session_id,
            model=m.model_name,
            state=m.state.value,
            turn_count=m.turn_count,
            total_prompt_tokens=m.total_prompt_tokens,
            total_completion_tokens=m.total_completion_tokens,
            total_tokens=m.total_tokens,
            created_at=m.created_at,
            last_active=m.last_active,
            memory_bytes=mgr.get_session_memory(m.session_id),
        ))
    return SessionListResponse(
        sessions=sessions,
        total_active_memory=mgr.get_active_memory(),
    )


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session details."""
    mgr = _mgr()
    m = mgr.get_session(session_id)
    if m is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return SessionInfo(
        session_id=m.session_id,
        model=m.model_name,
        state=m.state.value,
        turn_count=m.turn_count,
        total_prompt_tokens=m.total_prompt_tokens,
        total_completion_tokens=m.total_completion_tokens,
        total_tokens=m.total_tokens,
        created_at=m.created_at,
        last_active=m.last_active,
        memory_bytes=mgr.get_session_memory(m.session_id),
    )


@router.post("/{session_id}/chat")
async def session_chat(
    session_id: str,
    request: SessionChatRequest,
    http_request: FastAPIRequest,
):
    """Chat within a session. KV cache is retained between turns."""
    mgr = _mgr()
    manifest = mgr.get_session(session_id)
    if manifest is None:
        raise HTTPException(404, f"Session {session_id} not found")
    if manifest.state.value == "parked":
        raise HTTPException(
            409, "Session is parked. POST /v1/sessions/{id}/resume first."
        )
    if manifest.state.value in ("parking", "resuming"):
        raise HTTPException(
            409, f"Session is {manifest.state.value}. Try again shortly."
        )

    # Acquire per-session lock (sequential turns)
    lock = mgr.acquire_session_lock(session_id)
    try:
        return await _do_session_chat(
            mgr, manifest, request, http_request
        )
    finally:
        lock.release()


async def _do_session_chat(mgr, manifest, request, http_request):
    """Internal session chat handler."""
    # Get engine for the session's model
    engine = await _get_engine_for_model(manifest.model_name)

    # Extract messages
    from .utils import extract_text_content
    messages = extract_text_content(request.messages)

    # Apply chat template to get prompt
    chat_template_kwargs = {}
    if request.tools:
        chat_template_kwargs["tools"] = request.tools
    prompt = engine._apply_chat_template(
        messages, **chat_template_kwargs
    )

    # Build kwargs for engine
    chat_kwargs = {}
    if request.temperature is not None:
        chat_kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        chat_kwargs["top_p"] = request.top_p
    if request.top_k is not None:
        chat_kwargs["top_k"] = request.top_k
    if request.min_p is not None:
        chat_kwargs["min_p"] = request.min_p
    if request.max_tokens is not None:
        chat_kwargs["max_tokens"] = request.max_tokens
    if request.repetition_penalty is not None:
        chat_kwargs["repetition_penalty"] = request.repetition_penalty
    if request.presence_penalty is not None:
        chat_kwargs["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        chat_kwargs["frequency_penalty"] = request.frequency_penalty
    if request.stop:
        chat_kwargs["stop"] = request.stop

    # Inject session_id so scheduler retains KV
    chat_kwargs["session_id"] = manifest.session_id

    if request.stream:
        return StreamingResponse(
            _stream_session_chat(engine, prompt, chat_kwargs, manifest, mgr, messages),
            media_type="text/event-stream",
        )

    # Non-streaming
    output = await engine.generate(prompt=prompt, **chat_kwargs)

    # Update session messages
    updated_messages = list(messages) + [
        {"role": "assistant", "content": output.text}
    ]
    # Note: manifest is updated by scheduler via update_after_generation
    # We just update messages here for introspection
    manifest.messages = updated_messages

    # Re-read manifest for updated stats
    m = mgr.get_session(manifest.session_id)
    return SessionChatResponse(
        session_id=manifest.session_id,
        content=output.text,
        finish_reason=output.finish_reason,
        usage={
            "prompt_tokens": output.prompt_tokens,
            "completion_tokens": output.completion_tokens,
            "total_tokens": output.prompt_tokens + output.completion_tokens,
        },
        turn=m.turn_count if m else manifest.turn_count,
        cached_tokens=output.cached_tokens if hasattr(output, "cached_tokens") else 0,
    )


async def _stream_session_chat(
    engine, prompt, chat_kwargs, manifest, mgr, messages
):
    """SSE streaming for session chat."""
    import json as _json

    full_text = ""
    try:
        async for chunk in engine.stream_generate(prompt=prompt, **chat_kwargs):
            full_text += chunk.text
            data = {
                "session_id": manifest.session_id,
                "content": chunk.text,
                "finished": chunk.finished,
            }
            if chunk.finished:
                data["finish_reason"] = chunk.finish_reason
                data["usage"] = {
                    "prompt_tokens": chunk.prompt_tokens,
                    "completion_tokens": chunk.completion_tokens,
                    "total_tokens": chunk.prompt_tokens + chunk.completion_tokens,
                }
                m = mgr.get_session(manifest.session_id)
                data["turn"] = m.turn_count if m else manifest.turn_count
                data["cached_tokens"] = (
                    chunk.cached_tokens if hasattr(chunk, "cached_tokens") else 0
                )
            yield f"data: {_json.dumps(data)}\n\n"

        # Update messages for introspection
        manifest.messages = list(messages) + [
            {"role": "assistant", "content": full_text}
        ]
    except Exception as e:
        logger.error(f"Session stream error: {e}")
        yield f"data: {_json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@router.post("/{session_id}/park")
async def park_session(session_id: str):
    """Park session: move KV from memory to SSD."""
    mgr = _mgr()
    manifest = mgr.get_session(session_id)
    if manifest is None:
        raise HTTPException(404, f"Session {session_id} not found")
    if manifest.state.value != "active":
        raise HTTPException(409, f"Session is {manifest.state.value}, cannot park")

    # Get the store_cache function from the scheduler
    engine = await _get_engine_for_model(manifest.model_name)
    scheduler = engine._engine.engine.scheduler
    if scheduler.block_aware_cache is None:
        raise HTTPException(500, "No cache backend configured")

    store_fn = scheduler.block_aware_cache.store_cache
    success = mgr.park_session(session_id, store_fn)
    if not success:
        raise HTTPException(500, "Park failed")

    return {"session_id": session_id, "state": "parked"}


@router.post("/{session_id}/resume")
async def resume_session(session_id: str):
    """Resume session: load KV from SSD to memory."""
    mgr = _mgr()
    manifest = mgr.get_session(session_id)
    if manifest is None:
        raise HTTPException(404, f"Session {session_id} not found")
    if manifest.state.value != "parked":
        raise HTTPException(409, f"Session is {manifest.state.value}, cannot resume")

    engine = await _get_engine_for_model(manifest.model_name)
    scheduler = engine._engine.engine.scheduler
    if scheduler.block_aware_cache is None:
        raise HTTPException(500, "No cache backend configured")

    success = mgr.resume_session(
        session_id,
        fetch_cache_fn=scheduler.block_aware_cache.fetch_cache,
        reconstruct_cache_fn=scheduler.block_aware_cache.reconstruct_cache,
        extract_cache_states_fn=scheduler._extract_cache_states,
    )
    if not success:
        raise HTTPException(500, "Resume failed")

    return {"session_id": session_id, "state": "active"}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free all resources."""
    mgr = _mgr()
    if not mgr.delete_session(session_id):
        raise HTTPException(404, f"Session {session_id} not found")
    return {"deleted": True, "session_id": session_id}
