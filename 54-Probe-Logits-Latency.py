"""
54-Probe-Logits-Latency.py - 最小复现：量化 llama_get_logits_ith 的延迟

背景：
    53-Profile-Breakdown.py 实测出 predictor 每次 get_logits_ith 要 9.14 ms，
    而 talker 只要 12.5 µs（同一函数、同一进程、同一后端），相差 730 倍。
    该项占总墙钟 85%。本脚本剥掉 TTS 管线，只保留
    「加载模型 → 建 ctx → 单 token decode → 取 logits」，
    通过变体对照定位真正的成因。

用法：
    .venv/Scripts/python 54-Probe-Logits-Latency.py --model-dir model-base
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


def bench(fn, reps: int = 300) -> tuple[float, float]:
    """返回 (单次秒, 首次秒)。"""
    t_first = None
    t0 = time.perf_counter()
    for i in range(reps):
        a = time.perf_counter()
        fn()
        dt = time.perf_counter() - a
        if i == 0:
            t_first = dt
    return (time.perf_counter() - t0) / reps, t_first


def probe(name: str, model, embd_dim: int, n_ctx: int, n_seq_max: int = 1,
          embeddings: bool = False, warm_steps: int = 1, reps: int = 300):
    """建 ctx、跑几步 decode，然后分别测 get_logits / get_logits_ith。"""
    from qwen3_tts_gguf.inference import llama

    print(f"\n--- {name} ---")
    print(f"    cfg: n_ctx={n_ctx} n_seq_max={n_seq_max} embeddings={embeddings} "
          f"embd_dim={embd_dim}")

    ctx = llama.LlamaContext(model, n_ctx=n_ctx, embeddings=embeddings,
                             n_seq_max=n_seq_max)
    batch = llama.LlamaBatch(max(2, n_ctx), embd_dim=embd_dim)

    emb = np.zeros((1, embd_dim), dtype=np.float32)
    emb[0, 0] = 0.01
    for s in range(warm_steps):
        batch.set_embd(emb, pos=s, seq_id=0)
        ctx.decode(batch)

    variants = {
        "get_logits_ith(-1)": lambda: ctx.get_logits_ith(-1),
        "get_logits_ith(0)": lambda: ctx.get_logits_ith(0),
        "get_logits()": lambda: ctx.get_logits(),
    }
    out = {}
    for label, fn in variants.items():
        try:
            fn()
        except Exception as e:
            print(f"    {label:22s} 不可用: {type(e).__name__}")
            continue
        avg, first = bench(fn, reps=reps)
        out[label] = avg
        tag = "  <<< 慢" if avg > 1e-3 else ""
        print(f"    {label:22s} 单次 {avg * 1e6:10.2f} µs   首次 {first * 1e6:10.2f} µs{tag}")

    # decode 与取 logits 的相对成本
    avg_dec, _ = bench(lambda: (batch.set_embd(emb, pos=0, seq_id=0), ctx.decode(batch)),
                       reps=min(50, reps))
    print(f"    {'(set_embd + decode)':22s} 单次 {avg_dec * 1e6:10.2f} µs")

    del ctx
    gc.collect()
    return out


def probe_interleave(talker, predictor, frames: int = 12, sub: int = 15):
    """
    复刻真实管线的交替模式：每帧先 1 次 talker decode，再 15 次
    (predictor 取 logits → decode)。对比"单 context 独占"与"双 context 交替"。
    """
    from qwen3_tts_gguf.inference import llama

    print("\n" + "=" * 78)
    print("交替复现：talker decode 与 predictor 取 logits 交错 (复刻真实管线)")
    print("=" * 78)

    tctx = llama.LlamaContext(talker, n_ctx=2048, embeddings=True)
    pctx = llama.LlamaContext(predictor, n_ctx=64, embeddings=False)
    tbatch = llama.LlamaBatch(2048, embd_dim=talker.n_embd)
    pbatch = llama.LlamaBatch(2, embd_dim=predictor.n_embd)

    temb = np.zeros((1, talker.n_embd), dtype=np.float32)
    pemb = np.zeros((1, predictor.n_embd), dtype=np.float32)
    temb[0, 0] = pemb[0, 0] = 0.01

    def round_seq(ith_times, dec_p_times, dec_t_times, t_first=False):
        for f in range(frames):
            if t_first:
                a = time.perf_counter()
                tbatch.set_embd(temb, pos=f, seq_id=0)
                tctx.decode(tbatch)
                dec_t_times.append(time.perf_counter() - a)
            for cs in range(sub):
                a = time.perf_counter()
                pctx.get_logits_ith(-1)
                ith_times.append(time.perf_counter() - a)
                a = time.perf_counter()
                pbatch.set_embd(pemb, pos=cs, seq_id=0)
                pctx.decode(pbatch)
                dec_p_times.append(time.perf_counter() - a)

    def stat(name, xs):
        if not xs:
            return
        xs_sorted = sorted(xs)
        print(f"    {name:34s} n={len(xs):5d}  均值 {np.mean(xs) * 1e6:9.2f} µs  "
              f"中位 {xs_sorted[len(xs) // 2] * 1e6:9.2f} µs  "
              f"最大 {max(xs) * 1e6:9.2f} µs")

    # A. predictor 独占（只有 predictor 在跑）
    ith, dp, dt = [], [], []
    round_seq(ith, dp, dt, t_first=False)
    print("  A. predictor 独占（无 talker 交错）:")
    stat("predictor.get_logits_ith(-1)", ith)
    stat("predictor decode", dp)

    # B. 双 context 交替
    ith2, dp2, dt2 = [], [], []
    round_seq(ith2, dp2, dt2, t_first=True)
    print("  B. talker + predictor 交替（真实管线模式）:")
    stat("predictor.get_logits_ith(-1)", ith2)
    stat("predictor decode", dp2)
    stat("talker decode", dt2)

    if ith and ith2:
        r = np.mean(ith2) / max(np.mean(ith), 1e-12)
        print(f"\n    → 交替让 get_logits_ith 变慢 {r:.1f} 倍")
        if r > 10:
            print("      ⇒ 慢速源自多 context 在同一 Vulkan 设备上的调度/同步，")
            print("        与 ONNX/DML 侧无关；换 ONNX 后端解决不了这个问题。")

    del tctx, pctx
    gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model-base")
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()

    from qwen3_tts_gguf.inference import llama
    from qwen3_tts_gguf.inference import logger  # noqa: F401  确保初始化

    d = ROOT / args.model_dir
    talker_p, predictor_p = d / "qwen3_tts_talker.q5_k.gguf", d / "qwen3_tts_predictor.q8_0.gguf"
    for p in (talker_p, predictor_p):
        if not p.exists():
            print(f"❌ 缺少 {p}")
            return 1

    print("=" * 78)
    print("llama_get_logits_ith 延迟最小复现")
    print("=" * 78)
    print(f"  bin 后端: {sorted(p.name for p in (ROOT / 'qwen3_tts_gguf/inference/bin').glob('ggml-*.dll') if not p.name.startswith('ggml-cpu'))}")

    predictor = llama.LlamaModel(str(predictor_p), n_gpu_layers=-1, use_gpu=True)
    talker = llama.LlamaModel(str(talker_p), n_gpu_layers=-1, use_gpu=True)

    results = {}
    # 与 talker.py / stream.py 完全一致的两种配置
    results["talker  (n_ctx=2048, emb=True)"] = probe(
        "talker 复刻配置", talker, talker.n_embd, n_ctx=2048, embeddings=True, reps=args.reps)
    results["predictor (n_ctx=64, emb=False)"] = probe(
        "predictor 复刻配置", predictor, predictor.n_embd, n_ctx=64, embeddings=False, reps=args.reps)

    # ---- 变体对照：逐个改一个变量，看哪个才是主因 ----
    print("\n" + "=" * 78)
    print("变体对照（predictor 模型，每次只改一个变量）")
    print("=" * 78)

    variants = [
        ("n_ctx=2048 (对齐 talker)", dict(n_ctx=2048, embeddings=False)),
        ("embeddings=True", dict(n_ctx=64, embeddings=True)),
        ("n_seq_max=4", dict(n_ctx=64, embeddings=False, n_seq_max=4)),
        ("n_ctx=2048 + emb=True", dict(n_ctx=2048, embeddings=True)),
    ]
    for label, kw in variants:
        try:
            r = probe(label, predictor, predictor.n_embd, reps=args.reps, **kw)
            tgt = r.get("get_logits_ith(-1)")
            if tgt is not None:
                print(f"    => get_logits_ith(-1) = {tgt * 1e6:.2f} µs"
                      f"{'   <<< 复现了慢速' if tgt > 1e-3 else ''}")
        except Exception as e:
            print(f"    变体 {label} 失败: {type(e).__name__}: {e}")

    probe_interleave(talker, predictor)

    print("\n" + "=" * 78)
    print("结论提示")
    print("=" * 78)
    print("  若某个变体把 µs 级拉回 µs 级 → 该项就是主因，改 context 参数即可。")
    print("  若所有变体都慢 → 与 n_ctx/embeddings 无关，属 llama.cpp 内部行为。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
