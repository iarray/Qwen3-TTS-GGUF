"""
59-Test-ServerApi-Flow.py - 用真实引擎复刻 ServerApi 的单请求流程，并拆出每请求固定开销

背景:
    ServerApi.py 的 /tts 每次请求都:
        1. engine.create_stream()              <- 新建 talker ctx(n_ctx=2048) + batch（分配显存）
        2. stream.set_voice(ref_audio/voice)   <- CodecEncoder + SpeakerEncoder（encoder.py 硬编码 CPU）
                                                  外加 _normalize 里的整段预解码求 final_state
        3. stream.clone(text)                  <- 真正的生成 + 波形解码
    第 1、2 步与文本长度无关，是纯固定开销；文本越短，它对 RTF 的污染越大。

    另外本脚本支持把 predictor 放到 CPU（58-Probe 实测 CPU 比 Vulkan 快 2.13x）。

用法:
    .venv/Scripts/python 59-Test-ServerApi-Flow.py --predictor-cpu --runs 2
    .venv/Scripts/python 59-Test-ServerApi-Flow.py                --runs 2 --mode wav
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TEXT = "今天天气真不错，我们一起去公园走走吧，听说樱花开得正好。"


def patch_predictor_to_cpu():
    """（已弃用）早期验证用：让 engine 在加载 predictor 时强制走 CPU。
    现已由 TTSEngine(predictor_use_gpu=False) 正式提供，保留仅为对照。"""
    from qwen3_tts_gguf.inference import llama

    orig = llama.LlamaModel.load_model

    def load_model(self, model_path, n_gpu_layers=-1, use_gpu=1):
        if "predictor" in str(model_path).lower():
            return orig(self, model_path, n_gpu_layers=0, use_gpu=False)
        return orig(self, model_path, n_gpu_layers=n_gpu_layers, use_gpu=use_gpu)

    llama.LlamaModel.load_model = load_model


def find_voice_wav() -> Path | None:
    wavs = sorted(ROOT.glob("output/**/*.wav"))
    return wavs[0] if wavs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model-base")
    ap.add_argument("--voice-json", default="output/elaborate/Vivian.json")
    ap.add_argument("--mode", choices=["json", "wav"], default="json",
                    help="json=GUI 那种已缓存音色；wav=ServerApi 那种现场提取")
    ap.add_argument("--predictor-cpu", action="store_true")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--text", default=TEXT)
    args = ap.parse_args()

    from qwen3_tts_gguf.inference import TTSEngine, TTSConfig
    from qwen3_tts_gguf.inference.schema.result import TTSResult

    print("=" * 88)
    print(f"ServerApi 单请求流程复刻   predictor={'CPU' if args.predictor_cpu else 'Vulkan'}  "
          f"voice={args.mode}  runs={args.runs}")
    print("=" * 88)

    t0 = time.perf_counter()
    engine = TTSEngine(model_dir=args.model_dir, onnx_provider="GPU", chunk_size=12,
                       verbose=False, subprocess_decoder=True,
                       predictor_use_gpu=(False if args.predictor_cpu else None))
    t_init = time.perf_counter() - t0
    if not engine:
        print("❌ 引擎未就绪")
        return 1
    print(f"  引擎初始化            {t_init:8.2f} s   (一次性，服务启动时付)")

    if args.mode == "wav":
        wav = find_voice_wav()
        if wav is None:
            print("❌ 找不到参考 wav")
            return 1
        voice_src = str(wav)
    else:
        voice_src = str(ROOT / args.voice_json)
        if not Path(voice_src).exists():
            print(f"❌ 找不到 {voice_src}")
            return 1

    cfg = TTSConfig(temperature=0.8, sub_temperature=0.8, seed=42, sub_seed=45,
                    streaming=False)

    print()
    print(f"  {'#':>2} | {'create_stream':>13} | {'set_voice':>10} | {'clone(生成+解码)':>16} "
          f"| {'音频':>7} | {'墙钟':>7} | {'RTF':>6} | {'纯生成 RTF':>10}")
    print("  " + "-" * 88)

    rows = []
    for i in range(args.runs):
        t_cs = time.perf_counter()
        stream = engine.create_stream()
        t_create = time.perf_counter() - t_cs
        if stream is None:
            print("❌ create_stream 失败")
            return 1

        t_sv = time.perf_counter()
        ok = stream.set_voice(voice_src, args.text)
        t_setvoice = time.perf_counter() - t_sv
        if not ok:
            print("❌ set_voice 失败")
            return 1

        t_cl = time.perf_counter()
        res = stream.clone(args.text, config=cfg)
        t_clone = time.perf_counter() - t_cl
        if res is None:
            print("❌ clone 失败")
            return 1

        audio_s = len(res.audio) / res.sample_rate if res.audio is not None else 0.0
        wall = t_create + t_setvoice + t_clone
        rtf = wall / audio_s if audio_s else 0.0
        gen_rtf = t_clone / audio_s if audio_s else 0.0
        rows.append((t_create, t_setvoice, t_clone, audio_s, wall, rtf, gen_rtf))
        print(f"  {i + 1:>2} | {t_create * 1e3:>11.1f}ms | {t_setvoice * 1e3:>8.1f}ms "
              f"| {t_clone:>14.2f}s | {audio_s:>6.2f}s | {wall:>6.2f}s | {rtf:>6.3f} "
              f"| {gen_rtf:>10.3f}")

    if rows:
        avg = np.mean(rows, axis=0)
        print("  " + "-" * 88)
        print(f"  平均 | {avg[0] * 1e3:>11.1f}ms | {avg[1] * 1e3:>8.1f}ms | "
              f"{avg[2]:>14.2f}s | {avg[3]:>6.2f}s | {avg[4]:>6.2f}s | {avg[5]:>6.3f} "
              f"| {avg[6]:>10.3f}")
        print()
        print(f"  ▶ 每请求固定开销 (create_stream + set_voice) = "
              f"{(avg[0] + avg[1]) * 1e3:.0f} ms，与文本长度无关")
        print(f"  ▶ 把它摊到 {avg[3]:.1f}s 音频上，就占 RTF "
              f"{(avg[0] + avg[1]) / avg[3]:.3f}")
        print(f"  ▶ 真正生成部分的 RTF = {avg[6]:.3f}")

    engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
