"""
56-Probe-Ctx-Params.py - A/B：predictor 的 9ms/步 到底由哪个 context 参数造成

已知（55-Probe-Decode-Scaling 实测，RX 6750 GRE / Vulkan）:
    predictor 每步:  decode 414 µs  +  sync 9253 µs   -> 每帧 157 ms -> RTF 1.97
    同一进程里 talker 的 get_logits_ith 只要 12.5 µs（53-Profile-Breakdown）
    两者差异的候选：n_ctx / n_batch / embeddings / 模型大小 / 词表大小

本脚本在 predictor 模型上逐个改一个 context 参数，测 decode 与 sync 的单次耗时。
另外把 talker 模型也测一遍作为参照（预期 ~10 µs 量级）。

用法:
    .venv/Scripts/python 56-Probe-Ctx-Params.py --frames 8
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


def run(model, frames: int, warm: int = 2, batch_cap: int = 2, **ctx_kw) -> dict:
    n_embd = model.n_embd
    n_vocab = llama.llama_vocab_n_tokens(model.vocab)
    ctx = llama.LlamaContext(model, **ctx_kw)
    batch = llama.LlamaBatch(batch_cap, embd_dim=n_embd)

    rng = np.random.default_rng(3)
    feed = (rng.standard_normal((16, n_embd)).astype(np.float32) * 0.02)

    acc = {"decode": 0.0, "sync": 0.0, "np": 0.0, "ndec": 0, "frame": 0.0, "clean": 0.0}
    rcs = set()

    def frame(rec: bool):
        t0 = time.perf_counter()
        ctx.clear_kv_cache()
        for cs in range(16):
            if rec:
                a = time.perf_counter()
            batch.set_embd(feed[cs].reshape(1, -1), pos=cs, seq_id=0)
            rcs.add(ctx.decode(batch))
            if rec:
                acc["decode"] += time.perf_counter() - a
                acc["ndec"] += 1
            if rec:
                a = time.perf_counter()
            ptr = ctx.get_logits_ith(-1)
            if rec:
                acc["sync"] += time.perf_counter() - a
            if rec:
                a = time.perf_counter()
            L = np.ctypeslib.as_array(ptr, shape=(n_vocab,))
            _ = float(L[cs * 4096 % n_vocab])
            if rec:
                acc["np"] += time.perf_counter() - a
        if rec:
            acc["frame"] += time.perf_counter() - t0

    for _ in range(warm):
        frame(False)
    for _ in range(frames):
        frame(True)

    del ctx, batch
    gc.collect()
    return {"acc": acc, "rc": sorted(rcs), "n_vocab": n_vocab, "frames": frames}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model-base")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    d = ROOT / args.model_dir
    pred = llama.LlamaModel(str(d / "qwen3_tts_predictor.q8_0.gguf"), n_gpu_layers=-1, use_gpu=True)
    talk = llama.LlamaModel(str(d / "qwen3_tts_talker.q5_k.gguf"), n_gpu_layers=-1, use_gpu=True)

    # predictor 现状：stream.py 里就是这样建的
    variants = [
        ("predictor 现状     ", pred, dict(n_ctx=64,   n_batch=64,   embeddings=False)),
        ("predictor nb=2048  ", pred, dict(n_ctx=64,   n_batch=2048, embeddings=False)),
        ("predictor ctx=2048 ", pred, dict(n_ctx=2048, n_batch=2048, embeddings=False)),
        ("predictor emb=True ", pred, dict(n_ctx=64,   n_batch=64,   embeddings=True)),
        ("predictor emb=T+big", pred, dict(n_ctx=2048, n_batch=2048, embeddings=True)),
        ("predictor no-flash ", pred, dict(n_ctx=64,   n_batch=64,   embeddings=False, flash_attn=False)),
        ("predictor no-kqv   ", pred, dict(n_ctx=64,   n_batch=64,   embeddings=False, offload_kqv=False)),
        ("talker 参照        ", talk, dict(n_ctx=2048, n_batch=2048, embeddings=True)),
    ]

    print("=" * 96)
    print("context 参数 A/B（每个变体 16 步/帧，B=1）")
    print("=" * 96)
    print(f"  {'变体':<20} | {'n_vocab':>7} | {'decode/次':>10} | {'sync/次':>11} "
          f"| {'numpy/次':>9} | {'每帧':>9} | rc")
    print("  " + "-" * 92)

    rows = []
    for name, model, kw in variants:
        try:
            r = run(model, args.frames, batch_cap=max(2, 16), **kw)
        except Exception as e:
            print(f"  {name:<20} | 失败: {type(e).__name__}: {e}")
            continue
        acc, n = r["acc"], r["acc"]["ndec"]
        per = {k: acc[k] / max(n, 1) for k in ("decode", "sync", "np")}
        frame_ms = acc["frame"] / r["frames"] * 1e3
        rc = ",".join(str(x) for x in r["rc"]) or "0"
        rows.append((name, r["n_vocab"], per, frame_ms))
        print(f"  {name:<20} | {r['n_vocab']:>7} | {per['decode'] * 1e6:>8.1f} µs "
              f"| {per['sync'] * 1e6:>9.1f} µs | {per['np'] * 1e6:>7.1f} µs "
              f"| {frame_ms:>7.2f} ms | {rc}")

    print()
    print("=" * 96)
    print("结论线索")
    print("=" * 96)
    print("  · 若把 embeddings 改成 True 后 sync 掉到 µs 级 -> 成本在「输出层 logits 回读」这条路径")
    print("  · 若把 n_ctx/n_batch 改成 2048 后变快 -> 与 n_batch 相关的图/缓冲尺寸问题")
    print("  · 若怎么改都是 ~9ms -> 与 context 参数无关，是 predictor 这张图的固有执行开销")
    print("  · talker 参照行的 sync 若也是 ~10 µs -> 两个模型在同一后端上的行为确实天差地别")
    return 0


if __name__ == "__main__":
    sys.exit(main())
