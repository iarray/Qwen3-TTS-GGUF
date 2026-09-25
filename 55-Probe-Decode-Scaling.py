"""
55-Probe-Decode-Scaling.py - 定位 predictor 每步 decode 的真实代价及随批大小的变化

背景（53-Profile-Breakdown 实测，单路 5.68s 音频）:
    predictor 每帧 15 步自回归中:
        ctx.decode()          386 µs   <- 只是"提交"
        get_logits_ith()     9141 µs   <- 真正的等待在这里（占全墙钟 85%）
    而 54-Probe-Logits-Latency 孤立测 get_logits_ith 只有 1.3 µs —— 因为那个探针
    在计时循环里从不 decode，也就从不产生需要等待的 GPU 工作，量到的是空转。
    本脚本补上缺的那一环：每次 decode 后**强制读回 logits**（触发真实同步）。

目的（回答一个问题）:
    单次「decode + 同步」的耗时是否随批大小 B 增长？
        与 B 无关  -> 固定开销（启动/同步/权重读），批量可摊薄
                      => 这正好解释 GUI 批量 RTF 0.334 vs 单路 RTF ~2.0
        随 B 线性  -> 算力瓶颈，批量无收益

读数含义:
    decode   = 提交一次 llama_decode 的墙钟
    sync     = decode 后第一次拿 logits 指针的墙钟（含 D2H 与设备同步）
    numpy    = 包成 ndarray 并真实触碰一个元素
    fill/clear = 组 batch 与清 KV 的 host 开销

用法:
    .venv/Scripts/python 55-Probe-Decode-Scaling.py
    .venv/Scripts/python 55-Probe-Decode-Scaling.py --batches 1 4 16 --frames 20
"""
from __future__ import annotations

import argparse
import ctypes
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

if not globals().get("_BOUND"):
    if llama.llama_decode is None:
        llama.bind_llama_lib()
    _BOUND = True


# ----------------------------------------------------------------------------
# 单路 predictor 的复刻（见 inference/predictor.py: predict_frame）
# 批量 predictor 的复刻（见 inference/batch.py: _predict_frames）
# 两者循环结构一致，仅 n_seq_max / batch 内容不同。
# ----------------------------------------------------------------------------
def run_predictor_frames(predictor, B: int, frames: int, warm: int = 2) -> dict:
    n_embd = predictor.n_embd
    n_vocab = llama.llama_vocab_n_tokens(predictor.vocab)

    # 与 batch.py 完全一致的 ctx / batch 规模
    n_ctx = max(64, B * 17)
    ctx = llama.LlamaContext(predictor, n_ctx=n_ctx, n_batch=n_ctx,
                             n_seq_max=B, embeddings=False)
    batch = llama.LlamaBatch(max(2, B * 2), embd_dim=n_embd, n_seq_max=B)

    rng = np.random.default_rng(1234)
    # emb[cs] 对应第 cs 个码本的 embedding 表取出的 (B, n_embd)
    emb = (rng.standard_normal((16, B, n_embd)).astype(np.float32) * 0.02)
    h = (rng.standard_normal((B, n_embd)).astype(np.float32) * 0.02)

    acc = {"clear": 0.0, "fill": 0.0, "decode": 0.0, "sync": 0.0, "numpy": 0.0,
           "frame": 0.0, "ndec": 0, "nsync": 0}
    rcs = set()

    def one_frame(rec: bool) -> None:
        t_f0 = time.perf_counter()

        if rec:
            a = time.perf_counter()
        ctx.clear_kv_cache()
        if rec:
            acc["clear"] += time.perf_counter() - a

        c_in = np.empty((B, 2, n_embd), dtype=np.float32)
        c_in[:, 0] = h
        c_in[:, 1] = emb[0]

        if rec:
            a = time.perf_counter()
        last = batch.set_embd_multi([(c_in[i], 0, i) for i in range(B)])
        if rec:
            acc["fill"] += time.perf_counter() - a

        if rec:
            a = time.perf_counter()
        rcs.add(ctx.decode(batch))
        if rec:
            acc["decode"] += time.perf_counter() - a
            acc["ndec"] += 1

        # prefill 之后的 logits 在真实代码里也会被采样消费，这里同样读回
        if rec:
            a = time.perf_counter()
        ptr = ctx.get_logits_ith(last[-1])
        if rec:
            acc["sync"] += time.perf_counter() - a
            acc["nsync"] += 1
        if rec:
            a = time.perf_counter()
        L = np.ctypeslib.as_array(ptr, shape=(n_vocab,))
        _ = float(L[0])
        if rec:
            acc["numpy"] += time.perf_counter() - a

        for cs in range(1, 16):
            feed = emb[cs]

            if rec:
                a = time.perf_counter()
            last = batch.set_embd_multi([(feed[i], cs + 1, i) for i in range(B)])
            if rec:
                acc["fill"] += time.perf_counter() - a

            if rec:
                a = time.perf_counter()
            rcs.add(ctx.decode(batch))
            if rec:
                acc["decode"] += time.perf_counter() - a
                acc["ndec"] += 1

            if rec:
                a = time.perf_counter()
            ptr = ctx.get_logits_ith(last[-1])
            if rec:
                acc["sync"] += time.perf_counter() - a
                acc["nsync"] += 1

            if rec:
                a = time.perf_counter()
            L = np.ctypeslib.as_array(ptr, shape=(n_vocab,))
            _ = float(L[(cs - 1) * 2048])
            if rec:
                acc["numpy"] += time.perf_counter() - a

            if cs < 15:
                continue

        if rec:
            acc["frame"] += time.perf_counter() - t_f0

    for _ in range(warm):
        one_frame(False)
    for _ in range(frames):
        one_frame(True)

    del ctx, batch
    gc.collect()

    rcs.discard(0)
    return {"acc": acc, "bad_rc": sorted(rcs), "n_ctx": n_ctx, "n_vocab": n_vocab}


# ----------------------------------------------------------------------------
# 流水线测试：连续 K 次 decode 不读 logits，最后只读一次
#   若 K 次的总耗时 ≈ 单次 (decode+sync)，说明 GPU 在背靠背执行，
#   "sync" 那一项量到的是「等 GPU 把队列做完」，而不是每次 9ms 的硬件往返。
# ----------------------------------------------------------------------------
def run_pipeline_depth(predictor, B: int, K: int, frames: int) -> dict:
    n_embd = predictor.n_embd
    n_vocab = llama.llama_vocab_n_tokens(predictor.vocab)
    n_ctx = max(64, B * 17 + K)
    ctx = llama.LlamaContext(predictor, n_ctx=n_ctx, n_batch=n_ctx,
                             n_seq_max=B, embeddings=False)
    batch = llama.LlamaBatch(max(2, B * 2), embd_dim=n_embd, n_seq_max=B)

    rng = np.random.default_rng(7)
    feed = (rng.standard_normal((B, n_embd)).astype(np.float32) * 0.02)

    t_dec = t_sync = 0.0
    for _ in range(2):  # warm
        ctx.clear_kv_cache()
        for k in range(K):
            last = batch.set_embd_multi([(feed[i], k, i) for i in range(B)])
            ctx.decode(batch)
        np.ctypeslib.as_array(ctx.get_logits_ith(last[-1]), shape=(n_vocab,))[0]

    for _ in range(frames):
        ctx.clear_kv_cache()
        a = time.perf_counter()
        for k in range(K):
            last = batch.set_embd_multi([(feed[i], k, i) for i in range(B)])
            ctx.decode(batch)
        t_dec += time.perf_counter() - a

        a = time.perf_counter()
        L = np.ctypeslib.as_array(ctx.get_logits_ith(last[-1]), shape=(n_vocab,))
        _ = float(L[0])
        t_sync += time.perf_counter() - a

    del ctx, batch
    gc.collect()
    return {"dec_total": t_dec, "sync_total": t_sync, "K": K, "frames": frames}


def us(x: float) -> str:
    return f"{x * 1e6:.1f} µs"


def ms(x: float) -> str:
    return f"{x * 1e3:.2f} ms"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model-base")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--vocab", type=int, default=0,
                    help="覆盖 vocab 规模（用于模拟采样掩码成本），0=用真实值")
    args = ap.parse_args()

    path = ROOT / args.model_dir / "qwen3_tts_predictor.q8_0.gguf"
    if not path.exists():
        print(f"❌ 缺少 {path}")
        return 1

    print("=" * 88)
    print("Predictor 单步 decode / 同步 代价 vs 批大小")
    print("=" * 88)

    t_load = time.perf_counter()
    predictor = llama.LlamaModel(str(path), n_gpu_layers=-1, use_gpu=True)
    print(f"  模型: {path.name}  n_embd={predictor.n_embd}  "
          f"n_vocab={llama.llama_vocab_n_tokens(predictor.vocab)}  "
          f"(加载 {time.perf_counter() - t_load:.1f}s)")
    print()
    print("  B   | decode/次 |  sync/次  | numpy/次 | 每帧合计 | 每帧 decode 次数 | rc")
    print("  ----+-----------+-----------+----------+----------+------------------+----")

    rows = []
    for B in args.batches:
        r = run_predictor_frames(predictor, B, args.frames)
        acc, ndec, nsync, f = r["acc"], r["acc"]["ndec"], r["acc"]["nsync"], args.frames
        per_dec = acc["decode"] / max(ndec, 1)
        per_sync = acc["sync"] / max(nsync, 1)
        per_np = acc["numpy"] / max(nsync, 1)
        per_frame = acc["frame"] / f
        rows.append({"B": B, "dec": per_dec, "sync": per_sync, "np": per_np,
                     "frame": per_frame, "ndec": ndec / f, "ctx": r["n_ctx"],
                     "fill": acc["fill"] / f, "clear": acc["clear"] / f,
                     "bad_rc": r["bad_rc"]})
        bad = ",".join(str(x) for x in r["bad_rc"]) or "0"
        print(f"  {B:2d}  | {us(per_dec):>9} | {us(per_sync):>9} | {us(per_np):>8} "
              f"| {ms(per_frame):>8} | {ndec / f:>16.0f} | {bad}")

    print()
    print("=" * 88)
    print("折算：predictor 每音频秒的成本（12.5 帧/s）")
    print("=" * 88)
    print("     B  | 每帧 ms | 每音频秒 ms | 折合 RTF 贡献")
    print("  ------+---------+-------------+-------------")
    base = None
    for r in rows:
        per_audio = r["frame"] * 12.5
        if base is None:
            base = r["frame"]
        print(f"    {r['B']:3d} | {ms(r['frame']):>7} | {per_audio * 1e3:>9.0f} "
              f"| {per_audio:>12.3f}   ({base / r['frame']:.2f}x 相对 B=1)")

    # ---- 流水线深度 ----
    print()
    print("=" * 88)
    print("流水线深度：连续 K 次 decode 不读 logits，最后读一次")
    print("=" * 88)
    print("     B  |  K  | K 次 decode 合计 | 末尾一次 sync | 合计 | K 次若各自 sync 应为")
    print("  ------+-----+------------------+---------------+------+---------------------")
    for B in [1, 16]:
        for K in [1, 4, 15]:
            r = run_pipeline_depth(predictor, B, K, 8)
            n = r["frames"]
            dec = r["dec_total"] / n
            syn = r["sync_total"] / n
            ref = K * (rows[0]["dec"] + rows[0]["sync"]) if K > 1 else None
            ref_s = f"{ms(ref)} (B=1 外推)" if ref else "—"
            print(f"    {B:3d} | {K:3d} | {ms(dec):>16} | {ms(syn):>13} | "
                  f"{ms(dec + syn):>4} | {ref_s}")

    print()
    print("=" * 88)
    print("怎么读这张表")
    print("=" * 88)
    print("  1) sync/次 若不随 B 增长 → 每步的固定等待，批量可摊薄（GUI 批量快的根因）")
    print("     若随 B 线性增长 → 算力瓶颈，批量无用")
    print("  2) 每音频秒 ms / 1000 = 该部分的理论 RTF 下限")
    print("  3) 流水线一栏：K 次 decode 合计远小于 K×(decode+sync) → 说明 GPU 是背靠背")
    print("     执行的，sync 量到的是「等队列做完」，不是每次的硬件往返延迟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
