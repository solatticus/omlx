#!/usr/bin/env python3
"""
Benchmark: Stateless vs Session+TurboQuant

Runs a multi-turn conversation against a live omlx server and compares
wall-clock latency, cache hit rates, and memory usage between stateless
(/v1/chat/completions) and session (/v1/sessions/*/chat) modes.

Usage:
    python scripts/benchmark_sessions.py \
        --url http://localhost:8080 \
        --model Qwen3.5-VL-122B-A10B-4bit-CRACK \
        --turns 10 --max-tokens 128
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

# -----------------------------------------------------------------
# Conversation builder — deterministic, growing each turn
# -----------------------------------------------------------------

CONVERSATION = [
    ("user", "I'm building a web scraper in TypeScript with Bun. It fetches URLs from a queue, downloads HTML, extracts links, and stores results in SQLite. I want to add rate limiting per domain. What approach do you recommend?"),
    ("assistant", "Use a token bucket rate limiter per domain. Track available tokens and last request time in a Map keyed by domain. Consume a token before each request, refill based on elapsed time. Add Bun.sleep() delays when the bucket is empty."),
    ("user", "Show me the TypeScript implementation of the RateLimiter class."),
    ("assistant", "Here's a RateLimiter class with per-domain token tracking, configurable requests-per-second, and automatic token refill based on elapsed time since last request."),
    ("user", "Now add exponential backoff for 429 Too Many Requests responses."),
    ("assistant", "Added retry logic: on 429, wait 1s, then 2s, then 4s, max 3 retries. The backoff multiplier and max retries are configurable per domain."),
    ("user", "Add a per-domain request queue so concurrent scrape calls wait in line instead of all firing at once."),
    ("assistant", "Added an async queue using a promise chain per domain. Each domain gets its own queue. Concurrent requests for the same domain are serialized while different domains proceed in parallel."),
    ("user", "Write the integration test for the rate limiter. Test that two rapid requests to the same domain are properly spaced."),
    ("assistant", "Here's a test using bun:test that creates a RateLimiter at 2 req/s, fires two requests to the same domain, and asserts the second was delayed by at least 450ms."),
    ("user", "The test passes. Now add a metrics endpoint that shows requests per domain, average latency, and 429 retry counts."),
    ("assistant", "Added a getMetrics() method that returns per-domain stats including total requests, average response time, retry count, and current queue depth."),
    ("user", "Refactor the rate limiter into its own module with proper exports and add JSDoc comments."),
    ("assistant", "Moved RateLimiter to src/rate-limiter.ts with named exports for the class, config type, and metrics type. Added JSDoc on all public methods."),
    ("user", "One more thing — add a global concurrent connection limit across all domains. Default 50."),
    ("assistant", "Added a semaphore-style global concurrency limiter. The constructor takes maxGlobalConcurrency (default 50). Each request acquires a slot before proceeding, releases on completion or error."),
    ("user", "Write a README section documenting the rate limiter configuration options."),
]


def build_messages(turn):
    """Build message list for turn N (1-indexed). Grows each turn."""
    messages = []
    for i in range(min(turn * 2 - 1, len(CONVERSATION))):
        role, content = CONVERSATION[i]
        messages.append({"role": role, "content": content})
    return messages


def http_post(url, data, timeout=300):
    """Simple HTTP POST returning parsed JSON."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw, strict=False)
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        return {"error": raw, "status": e.code}


def http_post_stream_ttft(url, data, timeout=300):
    """HTTP POST with streaming — returns (ttft, total_time, last_data)."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft = None
    last_data = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: {"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    try:
                        last_data = json.loads(line[6:], strict=False)
                    except json.JSONDecodeError:
                        pass
                elif line == "data: [DONE]":
                    break
    except Exception as e:
        return ttft, time.perf_counter() - t0, {"error": str(e)}
    total = time.perf_counter() - t0
    return ttft or total, total, last_data


# -----------------------------------------------------------------
# Benchmark runners
# -----------------------------------------------------------------

def run_stateless(base_url, model, turns, max_tokens):
    """Benchmark stateless /v1/chat/completions."""
    results = []
    for turn in range(1, turns + 1):
        messages = build_messages(turn)
        t0 = time.perf_counter()
        resp = http_post(f"{base_url}/v1/chat/completions", {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        })
        elapsed = time.perf_counter() - t0

        usage = resp.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        results.append({
            "turn": turn,
            "elapsed": elapsed,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cached_tokens": cached,
        })
        pct = int(100 * cached / pt) if pt > 0 else 0
        print(f"  Stateless turn {turn:>2}: {elapsed:5.2f}s  prompt={pt:>4}  cached={cached:>4}  hit={pct}%")

    return results


def run_session(base_url, model, turns, max_tokens):
    """Benchmark session /v1/sessions/*/chat with streaming TTFT."""
    # Create session
    resp = http_post(f"{base_url}/v1/sessions", {"model": model})
    sid = resp.get("session_id")
    if not sid:
        print(f"  FAILED to create session: {resp}")
        return [], 0, sid

    results = []
    for turn in range(1, turns + 1):
        messages = build_messages(turn)

        ttft, elapsed, last = http_post_stream_ttft(
            f"{base_url}/v1/sessions/{sid}/chat",
            {"messages": messages, "max_tokens": max_tokens, "stream": True},
        )

        usage = last.get("usage", {}) if last else {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        cached = last.get("cached_tokens", 0) if last else 0

        results.append({
            "turn": turn,
            "ttft": ttft,
            "elapsed": elapsed,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cached_tokens": cached,
        })
        pct = int(100 * cached / pt) if pt > 0 else 0
        print(f"  Session  turn {turn:>2}: {elapsed:5.2f}s  ttft={ttft:5.3f}s  prompt={pt:>4}  cached={cached:>4}  hit={pct}%")

    # Get memory
    info = http_post(f"{base_url}/v1/sessions/{sid}", {}) if False else None
    # GET not POST for info
    try:
        req = urllib.request.Request(f"{base_url}/v1/sessions/{sid}")
        with urllib.request.urlopen(req, timeout=10) as r:
            info = json.loads(r.read().decode())
    except Exception:
        info = {}

    mem = info.get("memory_bytes", 0)
    return results, mem, sid


def run_park_resume(base_url, sid):
    """Time park and resume."""
    t0 = time.perf_counter()
    http_post(f"{base_url}/v1/sessions/{sid}/park", {})
    park_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    http_post(f"{base_url}/v1/sessions/{sid}/resume", {})
    resume_time = time.perf_counter() - t0

    return park_time, resume_time


# -----------------------------------------------------------------
# Output
# -----------------------------------------------------------------

def print_comparison(stateless, session, session_mem, park_time, resume_time):
    """Print formatted comparison table."""
    turns = len(stateless)

    print()
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print()
    print(f"  {'Turn':>4}  {'--- Stateless ---':^22}  {'--- Session + TQ ---':^26}")
    print(f"  {'':>4}  {'time':>6} {'prompt':>6} {'cache':>5} {'hit':>4}  {'time':>6} {'ttft':>6} {'prompt':>6} {'cache':>5} {'hit':>4}")
    print(f"  {'----':>4}  {'------':>6} {'------':>6} {'-----':>5} {'----':>4}  {'------':>6} {'------':>6} {'------':>6} {'-----':>5} {'----':>4}")

    total_stateless = 0
    total_session = 0

    for i in range(turns):
        s = stateless[i]
        ss = session[i] if i < len(session) else {}

        s_pct = int(100 * s.get("cached_tokens", 0) / s["prompt_tokens"]) if s.get("prompt_tokens") else 0
        ss_pct = int(100 * ss.get("cached_tokens", 0) / ss.get("prompt_tokens", 1)) if ss.get("prompt_tokens") else 0

        total_stateless += s.get("elapsed", 0)
        total_session += ss.get("elapsed", 0)

        print(
            f"  {s['turn']:>4}  "
            f"{s.get('elapsed', 0):5.2f}s {s.get('prompt_tokens', 0):>6} {s.get('cached_tokens', 0):>5} {s_pct:>3}%  "
            f"{ss.get('elapsed', 0):5.2f}s {ss.get('ttft', 0):5.3f}s {ss.get('prompt_tokens', 0):>6} {ss.get('cached_tokens', 0):>5} {ss_pct:>3}%"
        )

    print(f"  {'----':>4}  {'------':>6} {'------':>6} {'-----':>5} {'----':>4}  {'------':>6} {'------':>6} {'------':>6} {'-----':>5} {'----':>4}")
    savings = ((total_stateless - total_session) / total_stateless * 100) if total_stateless > 0 else 0
    print(f"  Total: {total_stateless:6.1f}s{' ' * 20}  {total_session:6.1f}s  ({savings:+.0f}%)")
    print()
    print(f"  Session memory (TurboQuant compressed): {session_mem / (1024*1024):.1f} MB")
    if park_time > 0:
        print(f"  Park: {park_time:.2f}s  |  Resume: {resume_time:.2f}s")
    print()


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: stateless vs session+TurboQuant"
    )
    parser.add_argument("--url", default="http://localhost:8080", help="omlx server URL")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--turns", type=int, default=8, help="Number of conversation turns")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per turn")
    parser.add_argument("--output", help="JSON output file path")
    parser.add_argument("--skip-stateless", action="store_true", help="Skip stateless benchmark")
    parser.add_argument("--skip-park", action="store_true", help="Skip park/resume test")
    args = parser.parse_args()

    print(f"Session Benchmark — {args.turns}-turn conversation, max_tokens={args.max_tokens}")
    print(f"Model: {args.model}")
    print(f"Server: {args.url}")
    print()

    # Stateless benchmark
    stateless_results = []
    if not args.skip_stateless:
        print("Running stateless benchmark...")
        stateless_results = run_stateless(args.url, args.model, args.turns, args.max_tokens)
        print()

    # Session benchmark
    print("Running session benchmark (with TurboQuant)...")
    session_results, session_mem, sid = run_session(args.url, args.model, args.turns, args.max_tokens)
    print()

    # Park/resume
    park_time = resume_time = 0
    if sid and not args.skip_park:
        print("Testing park/resume...")
        park_time, resume_time = run_park_resume(args.url, sid)
        print(f"  Park: {park_time:.2f}s  Resume: {resume_time:.2f}s")
        print()

    # Results
    if stateless_results and session_results:
        print_comparison(stateless_results, session_results, session_mem, park_time, resume_time)

    # Cleanup
    if sid:
        try:
            req = urllib.request.Request(
                f"{args.url}/v1/sessions/{sid}", method="DELETE"
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    # JSON output
    if args.output:
        data = {
            "model": args.model,
            "turns": args.turns,
            "max_tokens": args.max_tokens,
            "stateless": stateless_results,
            "session": session_results,
            "session_memory_bytes": session_mem,
            "park_time": park_time,
            "resume_time": resume_time,
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
