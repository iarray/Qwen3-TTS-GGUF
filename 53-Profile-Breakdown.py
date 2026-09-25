"""
53-Profile-Breakdown.py - 全链路耗时归因与跨模块搬运量量化

把一次合成分解成三类成本，用来判断"瓶颈在哪、要不要统一推理后端"：

  1. GPU 计算    —— llama.cpp 的 llm_decode（Vulkan/CUDA）、ONNX 的 decoder/sampler
  2. host 侧胶水 —— 全词表掩码、embedding 查表、numpy 求和、进程间队列
  3. 跨模块搬运  —— set_embd / get_embeddings / ONNX 状态张量往返

实现方式：运行期猴补丁（monkey-patch）核心方法，不改动 qwen3_tts_gguf 任何生产代码。

用法：
    .venv/Scripts/python 53-Profile-Breakdown.py                    # 默认 batch 模式
    .venv/Scripts/python 53-Profile-Breakdown.py --streaming        # 流式模式
    .venv/Scripts/python 53-Profile-Breakdown.py --tag cuda --onnx GPU
    .venv/Scripts/python 53-Profile-Breakdown.py --compare dml-batch cuda-batch

结果保存到 output/profile/<tag>.json，供 --compare 对比。
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / "output" / "profile"


# ---------------------------------------------------------------------------
# 输出工具（中文按双宽对齐）
# ---------------------------------------------------------------------------

def _setup_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int, right: bool = False) -> str:
    s = str(s)
    n = _w(s)
    if n >= width:
        return s
    return (" " * (width - n) + s) if right else (s + " " * (width - n))


_BUFFER: list[str] = []


def say(*args) -> None:
    """同时打印到终端并记入报告缓冲（终端编码可能不可靠，报告另存 UTF-8 文本）。"""
    s = " ".join(str(a) for a in args) if args else ""
    print(s)
    _BUFFER.append(s)


def banner(title: str) -> None:
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def table(headers: list[tuple[str, int, bool]], rows: list[list[str]]) -> None:
    """headers: [(标题, 宽度, 右对齐)]"""
    say("  " + "  ".join(pad(h, w, r) for h, w, r in headers))
    say("  " + "  ".join("-" * _w(h) for h, _, _ in headers))
    for row in rows:
        say("  " + "  ".join(pad(c, w, r) for c, (_, w, r) in zip(row, headers)))


def ms(x: float) -> str:
    return f"{x * 1000:.2f}"


def us(x: float) -> str:
    return f"{x * 1e6:.2f}"


# ---------------------------------------------------------------------------
# 计时累加器
# ---------------------------------------------------------------------------

class Acc:
    __slots__ = ("n", "t", "mx")

    def __init__(self) -> None:
        self.n = 0
        self.t = 0.0
        self.mx = 0.0

    def add(self, dt: float) -> None:
        self.n += 1
        self.t += dt
        if dt > self.mx:
            self.mx = dt

    @property
    def avg(self) -> float:
        return self.t / self.n if self.n else 0.0

    def snapshot(self) -> dict:
        return {"n": self.n, "total": self.t, "avg": self.avg, "max": self.mx}


STATS: dict[str, Acc] = {}
PER_RUN: list[dict] = []
MASK_CALLS: list[tuple[int, int, int]] = []
VOCAB: dict[str, int] = {}
# 实验开关：跳过 Python 侧的取 logits + 全词表掩码，用于判定该项是否为真瓶颈。
# 会改变生成结果（掩码没了），只看耗时，不可用于质量评估。
NO_MASK = False


def rec(key: str, dt: float) -> None:
    a = STATS.get(key)
    if a is None:
        a = STATS[key] = Acc()
    a.add(dt)


def reset_stats() -> None:
    STATS.clear()
    MASK_CALLS.clear()
    VOCAB.clear()


# ---------------------------------------------------------------------------
# 猴补丁
# ---------------------------------------------------------------------------

def install_patches() -> None:
    from qwen3_tts_gguf.inference import llama
    from qwen3_tts_gguf.inference import assets as assets_mod
    from qwen3_tts_gguf.inference import encoder as encoder_mod
    from qwen3_tts_gguf.inference import talker as talker_mod
    from qwen3_tts_gguf.inference import predictor as predictor_mod

    # --- llama.cpp 图执行（真正的 GPU 前向）---
    orig_decode = llama.LlamaContext.decode

    def decode_wrapped(self, batch):
        tag = getattr(self, "_prof_tag", "llm")
        t0 = time.perf_counter()
        try:
            return orig_decode(self, batch)
        finally:
            rec(f"{tag}.llm_decode", time.perf_counter() - t0)

    llama.LlamaContext.decode = decode_wrapped

    # --- batch 填充（host -> llama.cpp 输入缓冲）---
    orig_set_embd = llama.LlamaBatch.set_embd

    def set_embd_wrapped(self, data, pos=0, seq_id=0):
        t0 = time.perf_counter()
        try:
            return orig_set_embd(self, data, pos=pos, seq_id=seq_id)
        finally:
            rec("marshal.set_embd", time.perf_counter() - t0)

    llama.LlamaBatch.set_embd = set_embd_wrapped

    # --- 取 logits 指针（GPU 后端的隐式 D2H 同步可能就藏在这里）---
    orig_ith = llama.LlamaContext.get_logits_ith

    def ith_wrapped(self, i: int):
        tag = getattr(self, "_prof_tag", "llm")
        t0 = time.perf_counter()
        try:
            return orig_ith(self, i)
        finally:
            rec(f"{tag}.get_logits_ith", time.perf_counter() - t0)

    llama.LlamaContext.get_logits_ith = ith_wrapped

    # --- 采样（含 CPU 侧全词表掩码）---
    orig_sample = llama.LlamaSampler.sample

    def sample_wrapped(self, ctx, idx=-1, limit_start=None, limit_end=None, allow_tokens=None):
        tag = getattr(ctx, "_prof_tag", "llm")
        if NO_MASK:
            # 丢掉掩码约束 → sample() 内部整段跳过 get_logits_ith + 掩码写
            limit_start = limit_end = None
            allow_tokens = None
        t0 = time.perf_counter()
        try:
            return orig_sample(self, ctx, idx=idx, limit_start=limit_start,
                               limit_end=limit_end, allow_tokens=allow_tokens)
        finally:
            rec(f"{tag}.sampler", time.perf_counter() - t0)
            if limit_start is not None or limit_end is not None:
                try:
                    n_vocab = llama.llama_vocab_n_tokens(ctx.model.vocab)
                    VOCAB[tag] = int(n_vocab)
                    MASK_CALLS.append((int(n_vocab), int(limit_start or 0),
                                       int(n_vocab if limit_end is None else limit_end)))
                except Exception:
                    pass

    llama.LlamaSampler.sample = sample_wrapped

    # --- codec embedding 查表 ---
    def make_lookup_patch(name: str):
        orig = getattr(assets_mod.AssetsManager, name)

        def wrapped(self, q_idx, code):
            t0 = time.perf_counter()
            try:
                return orig(self, q_idx, code)
            finally:
                rec("marshal.codec_lookup", time.perf_counter() - t0)

        return wrapped

    for _name in ("get_codec_embedding", "get_codec_embedding_1024"):
        setattr(assets_mod.AssetsManager, _name, make_lookup_patch(_name))

    # --- ONNX 编码器 ---
    def make_enc_patch(cls, name: str, key: str):
        orig = getattr(cls, name)

        def wrapped(self, *a, **kw):
            t0 = time.perf_counter()
            try:
                return orig(self, *a, **kw)
            finally:
                rec(key, time.perf_counter() - t0)

        return wrapped

    # 只包 encode_audio（真正的计算），不包外层的 encode()，避免重复计数
    encoder_mod.CodecEncoder.encode = make_enc_patch(
        encoder_mod.CodecEncoder, "encode", "encoder.codec")
    encoder_mod.SpeakerEncoder.encode_audio = make_enc_patch(
        encoder_mod.SpeakerEncoder, "encode_audio", "encoder.speaker")

    # --- 顶层阶段 ---
    talker_mod.TalkerPredictor.prefill = make_enc_patch(
        talker_mod.TalkerPredictor, "prefill", "stage.talker_prefill")
    talker_mod.TalkerPredictor.decode_step = make_enc_patch(
        talker_mod.TalkerPredictor, "decode_step", "stage.talker_step")
    predictor_mod.Predictor.predict_frame = make_enc_patch(
        predictor_mod.Predictor, "predict_frame", "stage.predictor_frame")


# ---------------------------------------------------------------------------
# 掩码成本估算
# ---------------------------------------------------------------------------

_MASK_CACHE: dict[tuple[int, int, int], float] = {}


def mask_cost(n_vocab: int, s: int, e: int, reps: int = 120) -> float:
    """复刻 LlamaSampler.sample 里的掩码写法，测单次成本。"""
    key = (n_vocab, s, e)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    logits = np.zeros(n_vocab, dtype=np.float32)
    for _ in range(8):
        m = np.ones(n_vocab, dtype=bool)
        m[s:e] = False
        logits[m] = -np.inf
    t0 = time.perf_counter()
    for _ in range(reps):
        m = np.ones(n_vocab, dtype=bool)
        m[s:e] = False
        logits[m] = -np.inf
    dt = (time.perf_counter() - t0) / reps
    _MASK_CACHE[key] = dt
    return dt


# ---------------------------------------------------------------------------
# 搬运基准
# ---------------------------------------------------------------------------

def bench_transfer(nbytes_list: list[int], reps: int = 400) -> list[dict]:
    out = []
    for nb in nbytes_list:
        src = np.zeros(nb, dtype=np.uint8)
        dst = np.zeros(nb, dtype=np.uint8)
        for _ in range(8):
            ctypes.memmove(dst.ctypes.data, src.ctypes.data, nb)
        t0 = time.perf_counter()
        for _ in range(reps):
            ctypes.memmove(dst.ctypes.data, src.ctypes.data, nb)
        dt = (time.perf_counter() - t0) / reps
        out.append({"bytes": nb, "sec": dt, "gbps": (nb / dt) / 1e9 if dt else 0.0})
    return out


# ---------------------------------------------------------------------------
# 解码器探针（进程内，绕过子进程以纯化计时）
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """识别 bin/ 下实际存在的 ggml 加速后端 DLL，用于 A/B 对比时确认换了后端。"""
    bin_dir = ROOT / "qwen3_tts_gguf" / "inference" / "bin"
    if not bin_dir.is_dir():
        return "(bin/ 不存在)"
    found = sorted(p.name.replace("ggml-", "").replace(".dll", "")
                   for p in bin_dir.glob("ggml-*.dll")
                   if not p.name.startswith("ggml-cpu"))
    return ", ".join(found) if found else "(仅 CPU)"


def decoder_probe(decoder_onnx: Path, provider: str, chunk_sizes: list[int]) -> dict:
    from qwen3_tts_gguf.inference.decoder import StatefulDecoder

    dec = StatefulDecoder(str(decoder_onnx), onnx_provider=provider, chunk_size=4096)
    result = {"provider": dec.active_provider, "dtype": dec.dtype.__name__, "chunks": []}

    for cs in chunk_sizes:
        state = dec.create_state(72)
        codes = np.zeros((cs, 16), dtype=np.int64)
        for _ in range(3):
            _, state = dec._decode(codes, state=dec.create_state(72), is_final=False)
        state = dec.create_state(72)
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            _, state = dec._decode(codes, state=state, is_final=False)
        dt = (time.perf_counter() - t0) / reps
        result["chunks"].append({
            "frames": cs,
            "sec": dt,
            "ms_per_frame": dt * 1000 / cs,
            "audio_sec": cs / 12.5,
            "rtf": dt / (cs / 12.5),
        })

    st = dec.create_state(72)
    nb = (st.pre_conv_history.nbytes + st.latent_buffer.nbytes
          + st.conv_history.nbytes + sum(a.nbytes for a in st.kv_cache))
    result["state_bytes"] = nb
    return result


# ---------------------------------------------------------------------------
# 单次运行
# ---------------------------------------------------------------------------

def profile_run(args) -> int:
    from qwen3_tts_gguf.inference import TTSEngine, TTSConfig
    from qwen3_tts_gguf.inference.utils.audio import load_audio

    tag = args.tag or f"{args.onnx.lower()}-{'stream' if args.streaming else 'batch'}"
    voice = ROOT / args.voice
    if not voice.exists():
        print(f"❌ 找不到音色文件: {voice}")
        return 1

    global NO_MASK
    _BUFFER.clear()
    NO_MASK = bool(getattr(args, "no_mask", False))
    banner(f"Qwen3-TTS 全链路性能剖析   (tag = {tag})")
    say(f"  model_dir={args.model_dir}  onnx={args.onnx}  streaming={args.streaming}  "
        f"chunk_size={args.chunk_size}  no_mask={NO_MASK}")
    say(f"  llama.cpp 后端 DLL: {detect_backend()}")
    if NO_MASK:
        say("  ⚠️ 实验模式：已跳过 Python 侧取 logits + 全词表掩码，耗时可比、结果不可比")

    install_patches()

    # ------------------------------------------------------------------
    # [1/5] host 搬运地板价：与 GPU 后端无关，先把"拷贝到底值多少"钉死
    # ------------------------------------------------------------------
    banner("[1/5] host 内存搬运基准（ctypes.memmove 地板价）")
    state_nb = 2 * 16 * 72 * 64 * 2 * 8   # KV 8 层 × 2(k,v) × 1×16×72×64 × fp16
    xfer = bench_transfer([2048 * 4, 1024 * 4, 230 * 1024, state_nb])
    say("    代表 llama 隐层(8KB) / codec 嵌入(4KB) / 解码器状态(2.3MB) 的单次搬运成本：")
    table([("字节数", 12, True), ("≈KB", 10, True), ("单次耗时", 14, True), ("等效带宽", 14, True)],
          [[f"{x['bytes']}", f"{x['bytes'] / 1024:.1f}", f"{us(x['sec'])} µs",
            f"{x['gbps']:.1f} GB/s"] for x in xfer])

    # ------------------------------------------------------------------
    # [2/5] 解码器探针：必须在引擎之前跑。
    #       主进程再造一个 DML 会话会与子进程解码器抢 DXGI 设备，
    #       直接触发"GPU 设备已移除 (GetDeviceRemovedReason)"。
    # ------------------------------------------------------------------
    dec_probe = None
    dec_onnx = ROOT / args.model_dir / "qwen3_tts_decoder.fp16.onnx"
    if args.no_dec_probe:
        say("\n    (已按 --no-dec-probe 跳过解码器探针)")
    elif not dec_onnx.exists():
        say(f"\n    ⚠️ 找不到 {dec_onnx.name}，跳过解码器探针")
    else:
        banner("[2/5] 解码器探针（进程内 ONNX，单会话独占）")
        try:
            dec_probe = decoder_probe(dec_onnx, args.onnx, sorted({1, args.chunk_size, 64}))
            say(f"    provider={dec_probe['provider']}  精度={dec_probe['dtype']}  "
                f"状态张量={dec_probe['state_bytes'] / 1024:.0f} KB")
            table([("帧/块", 8, True), ("音频秒", 10, True), ("单块耗时", 14, True),
                   ("每帧", 12, True), ("解码 RTF", 12, True)],
                  [[f"{c['frames']}", f"{c['audio_sec']:.2f}", f"{ms(c['sec'])} ms",
                    f"{c['ms_per_frame']:.2f} ms", f"{c['rtf']:.4f}"]
                   for c in dec_probe["chunks"]])
        except Exception as e:
            say(f"    ⚠️ 解码器探针失败（{type(e).__name__}）: {e}")
            say("       若为设备移除/DXGI 报错，即为多套 GPU 运行时冲突的实证。")
            dec_probe = None
        finally:
            gc.collect()

    banner(f"[3/5] 初始化引擎  (onnx={args.onnx}, subprocess_decoder=True)")
    engine = TTSEngine(model_dir=args.model_dir, onnx_provider=args.onnx,
                       chunk_size=args.chunk_size, verbose=False,
                       subprocess_decoder=True)
    if not engine:
        print("❌ 引擎未就绪（检查 model-dir 与 bin/ 下的 llama.cpp DLL）")
        return 1

    # 给两个 llama context 打标签，区分 talker / predictor
    stream = engine.create_stream()
    stream.talker_ctx._prof_tag = "talker"
    stream.predictor_ctx._prof_tag = "predictor"

    cfg = TTSConfig(max_steps=args.max_steps, streaming=args.streaming,
                    temperature=args.temperature, sub_temperature=0.6,
                    seed=args.seed, sub_seed=args.seed + 3)

    banner("[4/5] 预热（不计量）")
    voice_res = stream.set_voice(str(voice))
    if not voice_res:
        print("❌ 音色准备失败")
        engine.shutdown()
        return 1
    warm = stream.clone(args.text, config=cfg)
    if warm is None:
        print("❌ 合成失败")
        engine.shutdown()
        return 1
    say(f"    预热完成: 音频 {warm.codes.shape[0] / 12.5:.2f}s, {warm.codes.shape[0]} 帧")

    banner(f"正式测量（{args.runs} 次，取中位数样本为代表）")
    runs = []
    for i in range(args.runs):
        reset_stats()
        t_wall0 = time.perf_counter()
        res = stream.clone(args.text, config=cfg)
        wall = time.perf_counter() - t_wall0
        if res is None:
            print(f"    ❌ 第 {i + 1} 次失败，跳过")
            continue
        snap = {k: v.snapshot() for k, v in STATS.items()}
        masks = {}
        for mc in MASK_CALLS:
            masks[mc] = masks.get(mc, 0) + 1
        runs.append({
            "wall": wall,
            "frames": int(res.codes.shape[0]),
            "audio_sec": float(res.duration),
            "stats": snap,
            "vocab": dict(VOCAB),
            "mask_calls": {f"{k[0]}:{k[1]}-{k[2]}": v for k, v in masks.items()},
            "rtf_core": float(res.rtf),
            "first_chunk": float(res.stats.first_chunk_latency) if res.stats else 0.0,
            "first_audio": float(res.stats.first_audio_latency) if res.stats else 0.0,
        })
        say(f"    第 {i + 1} 次: 墙钟 {wall:.2f}s, 音频 {res.duration:.2f}s, "
            f"RTF {wall / res.duration if res.duration else 0:.3f}")

    if not runs:
        print("❌ 没有有效的测量结果")
        engine.shutdown()
        return 1

    # ---- 汇总（取中位数那一次作为代表）----
    runs_sorted = sorted(runs, key=lambda r: r["wall"])
    rep = runs_sorted[len(runs_sorted) // 2]
    say("")
    say(f"    代表样本: 墙钟 {rep['wall']:.2f}s / 音频 {rep['audio_sec']:.2f}s")

    # ---- 编码器探针 ----
    enc_probe = None
    wavs = sorted(ROOT.glob(args.enc_probe or "output/*.wav"))
    if wavs and engine.codec_encoder is not None:
        wav = load_audio(wavs[0])
        if wav is not None:
            reset_stats()
            engine.codec_encoder.encode(wav)
            engine.speaker_encoder.encode(wav)
            enc_probe = {
                "wav": wavs[0].name,
                "audio_sec": len(wav) / 24000,
                "stats": {k: v.snapshot() for k, v in STATS.items()},
            }
            banner("[5/5] 编码器探针（一次性，不在关键路径）")
            say(f"    素材: {wavs[0].name}  ({enc_probe['audio_sec']:.2f}s 音频)")
            table([("阶段", 26, False), ("次数", 8, True), ("耗时", 14, True)],
                  [[k, f"{v['n']}", f"{ms(v['total'])} ms"]
                   for k, v in enc_probe["stats"].items()])

    print_report(rep, runs, xfer, dec_probe, enc_probe, state_nb, cfg)

    # ---- 落盘 ----
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "meta": {
            "model_dir": args.model_dir,
            "onnx_provider": args.onnx,
            "streaming": args.streaming,
            "chunk_size": args.chunk_size,
            "text": args.text,
            "llama_backend": detect_backend(),
            "decoder_provider": dec_probe["provider"] if dec_probe else None,
        },
        "representative": rep,
        "all_runs": runs,
        "transfer": xfer,
        "decoder_probe": dec_probe,
        "encoder_probe": enc_probe,
    }
    out = PROFILE_DIR / f"{tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = PROFILE_DIR / f"{tag}.txt"
    txt.write_text("\n".join(_BUFFER) + "\n", encoding="utf-8")
    print(f"\n💾 结果已保存: {out.relative_to(ROOT)}")
    print(f"💾 报告已保存: {txt.relative_to(ROOT)}")

    engine.shutdown()
    return 0


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def print_report(rep, runs, xfer, dec_probe, enc_probe, state_nb, cfg) -> None:
    s = rep["stats"]
    audio = rep["audio_sec"]
    frames = rep["frames"]
    loop_wall = rep["wall"]

    def total(key: str) -> float:
        return s.get(key, {}).get("total", 0.0)

    def cnt(key: str) -> int:
        return s.get(key, {}).get("n", 0)

    banner("耗时归因")

    talker_sampler = total("talker.sampler")
    talker_llm = total("talker.llm_decode")
    talker_step = total("stage.talker_step")
    pred_frame = total("stage.predictor_frame")
    pred_sampler = total("predictor.sampler")
    pred_llm = total("predictor.llm_decode")
    set_embd = total("marshal.set_embd")
    lookup = total("marshal.codec_lookup")
    gen_loop = talker_sampler + pred_frame + talker_step
    other = loop_wall - gen_loop

    say("")
    say("  ▶ 顶层阶段（占墙钟比例）")
    table([("阶段", 30, False), ("次数", 8, True), ("总耗时", 12, True),
           ("每次", 12, True), ("占墙钟", 10, True)],
          [
              ["Talker 采样", f"{cnt('talker.sampler')}", f"{ms(talker_sampler)} ms",
               f"{us(s.get('talker.sampler', {}).get('avg', 0))} µs",
               f"{talker_sampler / loop_wall * 100:.1f}%"],
              ["Predictor 循环", f"{cnt('stage.predictor_frame')}", f"{ms(pred_frame)} ms",
               f"{ms(s.get('stage.predictor_frame', {}).get('avg', 0))} ms",
               f"{pred_frame / loop_wall * 100:.1f}%"],
              ["Talker 反馈步", f"{cnt('stage.talker_step')}", f"{ms(talker_step)} ms",
               f"{us(s.get('stage.talker_step', {}).get('avg', 0))} µs",
               f"{talker_step / loop_wall * 100:.1f}%"],
              ["未归因(胶水/队列/GC)", "—", f"{ms(other)} ms", "—",
               f"{other / loop_wall * 100:.1f}%"],
              ["墙钟合计", "—", f"{ms(loop_wall)} ms", "—", "100%"],
          ])

    say("")
    say("  ▶ Predictor 内部拆解（每帧 15 次自回归）")
    p_other = pred_frame - pred_sampler - pred_llm
    table([("阶段", 30, False), ("次数", 8, True), ("总耗时", 12, True),
           ("每次", 12, True), ("占本方块", 10, True)],
          [
              ["采样(含全词表掩码)", f"{cnt('predictor.sampler')}",
               f"{ms(pred_sampler)} ms",
               f"{us(s.get('predictor.sampler', {}).get('avg', 0))} µs",
               f"{pred_sampler / pred_frame * 100:.1f}%"],
              ["llama.cpp 图执行", f"{cnt('predictor.llm_decode')}",
               f"{ms(pred_llm)} ms",
               f"{us(s.get('predictor.llm_decode', {}).get('avg', 0))} µs",
               f"{pred_llm / pred_frame * 100:.1f}%"],
              ["余量(查表/求和/memmove)", "—", f"{ms(p_other)} ms", "—",
               f"{p_other / pred_frame * 100:.1f}%"],
              ["小计", "—", f"{ms(pred_frame)} ms", "—", "100%"],
          ])

    # 掩码估算
    est_mask = 0.0
    for key, n in rep["mask_calls"].items():
        nv_s, rng = key.split(":")
        a, b = (int(x) for x in rng.split("-"))
        est_mask += mask_cost(int(nv_s), a, b) * n
    total_samples = cnt("talker.sampler") + cnt("predictor.sampler")
    say("")
    say("  ▶ host 侧胶水（与 GPU 后端无关，换后端也省不掉）")
    table([("项目", 34, False), ("次数", 8, True), ("总耗时", 12, True), ("每次", 12, True)],
          [
              ["全词表掩码 (估算)", f"{total_samples}", f"{ms(est_mask)} ms",
               f"{us(est_mask / total_samples if total_samples else 0)} µs"],
              ["talker 取 logits 指针", f"{cnt('talker.get_logits_ith')}",
               f"{ms(total('talker.get_logits_ith'))} ms",
               f"{us(s.get('talker.get_logits_ith', {}).get('avg', 0))} µs"],
              ["predictor 取 logits 指针", f"{cnt('predictor.get_logits_ith')}",
               f"{ms(total('predictor.get_logits_ith'))} ms",
               f"{us(s.get('predictor.get_logits_ith', {}).get('avg', 0))} µs"],
              ["set_embd (含 memmove)", f"{cnt('marshal.set_embd')}", f"{ms(set_embd)} ms",
               f"{us(s.get('marshal.set_embd', {}).get('avg', 0))} µs"],
              ["codec embedding 查表", f"{cnt('marshal.codec_lookup')}", f"{ms(lookup)} ms",
               f"{us(s.get('marshal.codec_lookup', {}).get('avg', 0))} µs"],
          ])
    if VOCAB:
        say("    词表规模: " + "  ".join(f"{k}={v:,}" for k, v in sorted(VOCAB.items())))
    if total_samples:
        say(f"    掩码占全部采样耗时的 "
            f"{est_mask / (talker_sampler + pred_sampler) * 100:.1f}%，"
            f"折合每帧 {est_mask / max(frames, 1) * 1000:.2f} ms")

    # 采样内部残余 = 总采样 - 掩码 - 取指针，归给 llama_sampler_sample 本体
    for tag_name, samp_t, ith_t in (
        ("talker", talker_sampler, total("talker.get_logits_ith")),
        ("predictor", pred_sampler, total("predictor.get_logits_ith")),
    ):
        n_mask = None
        est = 0.0
        for k, v in rep["mask_calls"].items():
            nv_s, rng = k.split(":")
            if int(nv_s) == VOCAB.get(tag_name, -1):
                a, b = (int(x) for x in rng.split("-"))
                est += mask_cost(int(nv_s), a, b) * v
        resid = samp_t - est - ith_t
        say(f"    {tag_name} 采样残余(原生 sampler 本体): {ms(resid)} ms"
            f"  (占其采样耗时 {resid / samp_t * 100 if samp_t else 0:.1f}%)")

    if dec_probe:
        nb = dec_probe["state_bytes"]
        say("")
        say(f"  ▶ 解码器状态张量往返 (ONNX / {dec_probe['provider']})")
        table([("项目", 34, False), ("值", 18, True)],
              [
                  ["状态张量大小", f"{nb / 1024:.0f} KB"],
                  ["单次调用往返 (H2D+D2H)", f"{nb * 2 / 1024:.0f} KB"],
                  ["每秒音频的往返量", f"{nb * 2 * 12.5 / 1024 / 1024:.1f} MB/s"],
                  ["该往返的理论耗时", f"{us(2 * nb / (xfer[-1]['gbps'] * 1e9))} µs/次"
                                      f" (按 memmove 基准)"],
              ])

    say("")
    say("  ▶ 汇总")
    table([("指标", 30, False), ("值", 18, True)],
          [
              ["音频时长", f"{audio:.2f} s ({frames} 帧)"],
              ["推理墙钟 (含解码等待)", f"{loop_wall:.2f} s"],
              ["端到端 RTF", f"{loop_wall / audio if audio else 0:.3f}"],
              ["核心 RTF (不含解码渲染)", f"{rep['rtf_core']:.3f}"],
              ["首 chunk 延迟", f"{rep['first_chunk'] * 1000:.0f} ms"],
              ["首音延迟", f"{rep['first_audio'] * 1000:.0f} ms"],
              ["每帧音频的 host 开销", f"{(other + est_mask) / max(frames, 1) * 1000:.2f} ms "
                                       f"(帧预算 80 ms)"],
          ])


def compare_runs(tag_a: str, tag_b: str) -> int:
    pa, pb = PROFILE_DIR / f"{tag_a}.json", PROFILE_DIR / f"{tag_b}.json"
    for p in (pa, pb):
        if not p.exists():
            print(f"❌ 找不到 {p}")
            return 1
    a = json.loads(pa.read_text(encoding="utf-8"))
    b = json.loads(pb.read_text(encoding="utf-8"))

    banner(f"对比: {tag_a}  →  {tag_b}")
    ma, mb = a["meta"], b["meta"]
    print(f"  A: llama={ma.get('llama_backend')} onnx={ma['onnx_provider']} "
          f"streaming={ma['streaming']} decoder={ma.get('decoder_provider')}")
    print(f"  B: llama={mb.get('llama_backend')} onnx={mb['onnx_provider']} "
          f"streaming={mb['streaming']} decoder={mb.get('decoder_provider')}")
    print()

    keys = sorted(set(a["representative"]["stats"]) | set(b["representative"]["stats"]))
    rows = []
    for k in keys:
        sa = a["representative"]["stats"].get(k, {}).get("total", 0.0)
        sb = b["representative"]["stats"].get(k, {}).get("total", 0.0)
        if sa == 0 and sb == 0:
            continue
        delta = (sb - sa) / sa * 100 if sa else float("inf")
        rows.append([k, f"{ms(sa)}", f"{ms(sb)}",
                     "—" if sa == 0 else f"{delta:+.1f}%"])
    rows.sort(key=lambda r: -abs(float(r[2].split()[0]) - float(r[1].split()[0])))
    table([("阶段", 30, False), (f"{tag_a} (ms)", 14, True),
           (f"{tag_b} (ms)", 14, True), ("变化", 12, True)], rows)

    wa = a["representative"]["wall"] / a["representative"]["audio_sec"]
    wb = b["representative"]["wall"] / b["representative"]["audio_sec"]
    print(f"\n  端到端 RTF: {wa:.4f}  →  {wb:.4f}  ({(wb - wa) / wa * 100:+.1f}%)")

    da, db = a.get("decoder_probe"), b.get("decoder_probe")
    if da and db:
        pa_ = da["chunks"][0]["rtf"]
        pb_ = db["chunks"][0]["rtf"]
        print(f"  解码器 RTF: {pa_:.4f} ({da['provider']})  →  "
              f"{pb_:.4f} ({db['provider']})  ({(pb_ - pa_) / pa_ * 100:+.1f}%)")
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> int:
    _setup_stdout()
    ap = argparse.ArgumentParser(
        description="Qwen3-TTS 全链路耗时归因与跨模块搬运量量化")
    ap.add_argument("--model-dir", default="model-base")
    ap.add_argument("--voice", default="output/elaborate/Vivian.json",
                    help="音色锚点 (.json 或 .wav)")
    ap.add_argument("--text", default="今天天气真不错，我们一起去公园走走吧，听说樱花开得正好。")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--streaming", action="store_true", help="启用流式（默认 batch）")
    ap.add_argument("--onnx", default="GPU", help="ONNX provider: GPU / CUDA / DML / CPU")
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--tag", default=None, help="结果标签，默认按后端自动生成")
    ap.add_argument("--enc-probe", default="", help="编码器探针用的 wav glob，留空则扫 output/*.wav")
    ap.add_argument("--no-dec-probe", action="store_true", help="跳过多余的解码器探针")
    ap.add_argument("--no-mask", action="store_true",
                    help="实验：跳过 Python 侧取 logits + 全词表掩码，只看耗时（结果会变）")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="对比两次已保存的结果")
    args = ap.parse_args()

    if args.compare:
        return compare_runs(*args.compare)
    return profile_run(args)


if __name__ == "__main__":
    sys.exit(main())
