"""
57-Probe-Token-Scaling.py - 判定：每次 llama_decode 的代价是「固定开销」还是「随工作量增长」

已知:
    55-Probe-Decode-Scaling: predictor 每步 decode+sync ≈ 9.6 ms，
        且 B=1→16 时 sync 只从 9.25ms 涨到 14.6ms（工作量涨 16 倍）
    56-Probe-Ctx-Params:   改 n_ctx / n_batch / embeddings / flash_attn 全无效；
        talker(1006MB) 也是 6.6 ms/次，predictor(151MB) 9.6 ms/次

本脚本把**单次 decode 里的 token 数**从 1 扫到 512:
    若单次耗时几乎不随 token 数变化  -> 每次调用是"固定开销"（图启动/依赖链延迟）
                                         => 减少调用次数或合并调用才有收益
    若随 token 数近似线性             -> 是真实算力/带宽成本
                                         => 只能换后端或换硬件

并对比「1 次 16 token」与「16 次 1 token」，直接量出"分成 16 次调用"要多付多少。

用法:
    .venv/Scripts/python 57-Probe-Token-Scaling.py
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from qwen3_tts_gguf.inference import llama  # noqa: E402

if llama.llama_decode is None:
    llama.bind_llama_lib()


def bench_tokens(model, tokens: list[int], reps: int = 6, n_ctx: int = 1024) -> list[dict]:
    n_embd = model.n_embd
    n_vocab = llama.llama_vocab_n_tokens(model.vocab)
    ctx = llama.LlamaContext(model, n_ctx=n_ctx, n_batch=n_ctx, n_seq_max=1,
                             embeddings=False)
    batch = llama.LlamaBatch(n_ctx, embd_dim=n_embd)
    rng = np.random.default_rng(11)
    emb = (rng.standard_normal((max(tokens), n_embd)).astype(np.float32) * 0.02)

    def one(T: int) -> tuple[float, float]:
        ctx.clear_kv_cache()
        a = time.perf_counter()
        batch.set_embd(emb[:T], pos=0, seq_id=0)
        rc = ctx.decode(batch)
        t_dec = time.perf_counter() - a
        a = time.perf_counter()
        ptr = ctx.get_logits_ith(-1)
        t_syn = time.perf_counter() - a
        # 真实触碰，确保 logits 已可选地回读到 host
        _ = float(np.ctypeslib.as_array(ptr, shape=(n_vocab,))[-1])
        assert rc == 0, f"decode rc={rc} T={T}"
        return t_dec, t_syn

    out = []
    for T in tokens:
        for _ in range(2):  # warm
            one(T)
        ds, ss = [], []
        for _ in range(reps):
            d, s = one(T)
            ds.append(d)
            ss.append(s)
        out.append({"T": T, "dec": float(np.median(ds)), "sync": float(np.median(ss))})

    del ctx, batch
    gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model-base")
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    d = ROOT / args.model_dir

    print("=" * 92)
    print("单次 llama_decode 的代价 vs 调用内 token 数")
    print("=" * 92)

    for tag, fname in (("predictor q8_0 151MB", "qwen3_tts_predictor.q8_0.gguf"),
                       ("talker q5_k 1006MB", "qwen3_tts_talker.q5_k.gguf")):
        model = llama.LlamaModel(str(d / fname), n_gpu_layers=-1, use_gpu=True)
        print(f"\n--- {tag}  (n_embd={model.n_embd}, "
              f"n_vocab={llama.llama_vocab_n_tokens(model.vocab)}) ---")
        print(f"  {'token 数':>8} | {'decode':>11} | {'sync':>11} | {'合计':>11} "
              f"| {'每 token 边际':>14}")
        print("  " + "-" * 74)
        rows = bench_tokens(model, [1, 2, 4, 8, 16, 64, 256, 512], reps=args.reps)
        prev = None
        for r in rows:
            tot = r["dec"] + r["sync"]
            marg = ""
            if prev and r["T"] > prev["T"]:
                dp = (tot - (prev["dec"] + prev["sync"])) / (r["T"] - prev["T"])
                marg = f"{dp * 1e6:>9.2f} µs"
            print(f"  {r['T']:>8} | {r['dec'] * 1e6:>8.1f} µs | {r['sync'] * 1e6:>8.1f} µs "
                  f"| {tot * 1e6:>8.1f} µs | {marg}")
            prev = r

        t1 = rows[0]
        t512 = rows[-1]
        one_tok_call = (t1["dec"] + t1["sync"]) * 16
        one_call_16 = next(r for r in rows if r["T"] == 16)
        print(f"\n    16 次「1 token」调用 = {one_tok_call * 1e3:.2f} ms")
        print(f"    1  次「16 token」调用 = {(one_call_16['dec'] + one_call_16['sync']) * 1e3:.2f} ms")
        print(f"    → 拆成 16 次调用多付 "
              f"{(one_tok_call - (one_call_16['dec'] + one_call_16['sync'])) * 1e3:.2f} ms"
              f"（{one_tok_call / max(one_call_16['dec'] + one_call_16['sync'], 1e-9):.1f}x）")
        print(f"    512 token 单次 = {(t512['dec'] + t512['sync']) * 1e3:.2f} ms"
              f"  (是 1 token 的 {(t512['dec'] + t512['sync']) / (t1['dec'] + t1['sync']):.2f} 倍)")
        del model
        gc.collect()

    print()
    print("=" * 92)
    print("怎么读")
    print("=" * 92)
    print("  · 512 token 的耗时若只有 1 token 的 1~2 倍 → 每次调用是固定开销（图启动/依赖链）")
    print("    ⇒ 单路提速的唯一途径是「减少调用次数」或「一次调用塞更多路」（批量）")
    print("  · 若近似 512 倍 → 真实算力成本，只能换后端/硬件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
