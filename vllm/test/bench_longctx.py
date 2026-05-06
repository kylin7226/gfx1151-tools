"""Long-context (~25K token) test. Feeds the model a large synthesized
text passage and asks a comprehension question. Tests throughput at
long context AND answer quality.

Run: python3 test/bench_longctx.py
Output: test/bench_longctx_result.json
"""

from __future__ import annotations
import argparse
import json
import time
import urllib.request
import urllib.error

HOST = "http://127.0.0.1:8000"
MODEL = "Qwen3.6-27B-AWQ4"
TIMEOUT = 1800

# A long synthetic passage to exercise the context window.
# Repeated ~200 times to produce ~25K tokens of input.
PASSAGE = """
The unified memory architecture of the AMD Strix Halo platform enables large
language model inference that would traditionally require discrete GPUs with
dedicated VRAM. The iGPU shares the same physical DRAM as the CPU, which
eliminates the host-to-device transfer overhead that discrete GPUs incur.
For LLM inference, this enables loading models that exceed traditional GPU
VRAM budgets - a 27B parameter model in 4-bit AWQ quantization fits easily
within the 128 GB memory budget. The ROCm software stack provides the necessary
compiler support for RDNA 3.5 architecture targets, with attention backends
implemented via Triton JIT compilation at runtime. The vLLM inference engine
handles paged KV cache management, automatic chunked prefill, and OpenAI
compatible API serving. Profile caching reduces cold start time from roughly
9 minutes to approximately 95 seconds on subsequent restarts.
""".strip()

QUESTION = """
Based on the passage above, answer the following:

1. What advantage does unified memory provide for LLM inference compared to discrete GPUs?
2. How much memory does a 27B parameter AWQ-quantized model require?
3. What reduces cold start time from 9 minutes to ~95 seconds?

Be concise.
""".strip()


def post(path, body, timeout=TIMEOUT):
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()), time.perf_counter() - t0


def build_context() -> str:
    """Repeat the passage to build ~25K tokens of context."""
    parts = [PASSAGE] * 200
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    context = build_context()
    prompt = f"{context}\n\n{QUESTION}"
    chars = len(prompt)
    est_tokens = chars // 4
    print(f"Context: {chars:,} chars, ~{est_tokens:,} tokens (4 chars/token rough estimate)")
    print(f"Asking comprehension question, max_tokens={args.max_tokens}")
    print()

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }

    t0 = time.perf_counter()
    data, wall = post("/v1/chat/completions", body, timeout=TIMEOUT)
    t1 = time.perf_counter()

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    u = data.get("usage", {})
    pt = u.get("prompt_tokens") or 0
    ct = u.get("completion_tokens") or 0
    finish = data["choices"][0].get("finish_reason")

    decode_tps = ct / wall if wall else 0
    e2e_tps = ct / (t1 - t0) if (t1 - t0) else 0

    print(f"=== Long-context result ===")
    print(f"  prompt_tokens (actual): {pt}")
    print(f"  completion_tokens:      {ct}")
    print(f"  reasoning_chars:        {len(reasoning)}")
    print(f"  wall:                   {wall:.2f}s")
    print(f"  finish_reason:          {finish}")
    print(f"  decode t/s:             {decode_tps:.2f}")
    print(f"  e2e t/s:                {e2e_tps:.2f}")
    print()
    print("=== Answer (first 2000 chars) ===")
    print(content[:2000])

    import json as _json
    from pathlib import Path
    out = Path(__file__).parent / "bench_longctx_result.json"
    out.write_text(_json.dumps({
        "prompt_chars": chars,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_chars": len(reasoning),
        "wall_seconds": round(wall, 2),
        "decode_tps": round(decode_tps, 3),
        "e2e_tps": round(e2e_tps, 3),
        "finish_reason": finish,
        "reasoning": reasoning,
        "content": content,
    }, indent=2))
    print(f"\n=== Saved to {out} ===")


if __name__ == "__main__":
    main()
