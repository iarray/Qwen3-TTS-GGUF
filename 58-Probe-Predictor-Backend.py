"""
58-Probe-Predictor-Backend.py - 既然每步是「固定调用开销」，换后端能否绕开

已知（57-Probe-Token-Scaling）:
    predictor 单次 llama_decode 的代价几乎是常数:
        1 token -> 9.70 ms,  2 -> 9.84 ms,  16 -> 12.50 ms,  512 -> 64.39 ms
    把 16 次「1 token」合并成 1 次「16 token」:  155.22 ms -> 12.50 ms (省 12.4x)
    => 每帧 16 次调用 × ~9.6 ms 固定开销 = 155 ms / 80 ms 帧 -> 单路 RTF ≈ 1.9

本脚本对比 predictor 放在 Vulkan 与 CPU 上的每帧代价。
若 CPU 反而更快 -> 直接把 predictor 放 CPU 就是单路提速的捷径（talker 仍占 GPU）。

用法:
    .venv/Scripts/python 58-Probe-Predictor-Backend.py
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


def measure(model, frames: int, warm: int = 2, n_ctx: int = 64, n_threads=None) -> dict:
    """复刻 predictor.predict_frame 的每帧 16 次调用（1 次 2-token prefill + 15 次 1-token）"""
    n_embd = model.n_embd
    n_vocab = llama.llama_vocab_n_tokens(model.vocab)
    kw = dict(n_ctx=n_ctx, n_batch=n_ctx, n_seq_max=1, embeddings=False)
    if n_threads is not None:
        kw["n_threads"] = n_threads
    ctx = llama.LlamaContext(model, **kw)
    batch = llama.LlamaBatch(2, embd_dim=n_embd)

    rng = np.random.default_rng(5)
    emb = (rng.standard_normal((16, n_embd)).astype(np.float32) * 0.02)

    t_all = 0.0
    n_call = 0

    def frame(rec: bool) -> float:
        nonlocal t_all, n_call
        t0 = time.perf_counter()
        ctx.clear_kv_cache()
        # 与 predictor.py 一致：prefill 2 token 占 pos 0/1，后续每步起始 pos = cs + 1
        # （若沿用 pos=cs 会与上一位置重复，llama.cpp 直接 rc=-1 拒绝整批）
        for cs in range(16):
            if cs == 0:
                data, p = np.stack([emb[0], emb[0]], axis=0), 0
            else:
                data, p = emb[cs].reshape(1, -1), cs + 1
            batch.set_embd(data, pos=p, seq_id=0)
            rc = ctx.decode(batch)
            assert rc == 0, f"decode rc={rc}"
            ptr = ctx.get_logits_ith(-1)
            _ = float(np.ctypeslib.as_array(ptr, shape=(n_vocab,))[0])
            if rec:
                n_call += 1
        dt = time.perf_counter() - t0
        if rec:
            t_all += dt
        return dt

    for _ in range(warm):
        frame(False)
    for _ in range(frames):
        frame(True)

    del ctx, batch
    gc.collect()
    return {"frame": t_all / frames, "per_call": t_all / max(n_call, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model-base")
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    path = ROOT / args.model_dir / "qwen3_tts_predictor.q8_0.gguf"

    print("=" * 84)
    print("predictor 放 Vulkan 还是 CPU —— 每帧 16 次调用（1×2token + 15×1token）")
    print("=" * 84)
    print(f"  {'后端':<26} | {'每帧':>10} | {'每次调用':>10} | {'折合 RTF':>9}")
    print("  " + "-" * 70)

    rows = []
    for label, use_gpu, nthr in (("Vulkan (现状)", True, None),
                                 ("CPU", False, None)):
        m = llama.LlamaModel(str(path), n_gpu_layers=(-1 if use_gpu else 0), use_gpu=use_gpu)
        r = measure(m, args.frames)
        rtf = r["frame"] * 12.5
        rows.append((label, r, rtf))
        print(f"  {label:<26} | {r['frame'] * 1e3:>8.2f} ms | "
              f"{r['per_call'] * 1e6:>8.1f} µs | {rtf:>9.3f}")
        del m
        gc.collect()

    if len(rows) == 2:
        g, c = rows[0][1]["frame"], rows[1][1]["frame"]
        print()
        if c < g:
            print(f"  ✅ CPU 反而快 {g / c:.2f}x —— predictor 可以直接挪到 CPU，"
                  f"GPU 留给 talker/解码器")
        else:
            print(f"  ➖ CPU 慢 {c / g:.2f}x —— 保持 Vulkan")

    print()
    print("  参考：53-Profile-Breakdown 实测单路整机 RTF ≈ 2.02，其中 predictor 占 90.9%")
    print("        所以本表「折合 RTF」基本就是端到端 RTF 的下限")
    return 0


if __name__ == "__main__":
    sys.exit(main())
