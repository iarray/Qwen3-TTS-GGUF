from fastapi import FastAPI, Query
from fastapi.responses import Response
import io
import soundfile as sf
import os
import argparse
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

# 真实的 qwen3‑tts‑gguf 导入
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig
from qwen3_tts_gguf.inference.schema.result import TTSResult

app = FastAPI(title="Qwen3‑TTS‑GGUF Local API")

engine: TTSEngine | None = None
ROOT = Path(__file__).resolve().parent
MAX_TEXT_CHARS = 1000

# 音色锚点缓存：key = (参考音频绝对路径, mtime, 参考文本)
# 构建一次要跑 CodecEncoder + SpeakerEncoder + 一次完整预解码（求 final_state），
# 实测约 430 ms，且与请求文本无关，所以必须缓存，否则每个短请求都被它拖慢。
VOICE_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
VOICE_CACHE_MAX = 16


def convert_lang(s: str) -> str:
    """和原服务保持语言参数兼容"""
    s = (s or "").strip().lower()
    if s in ("zh", "chinese"):
        return "chinese"
    if s in ("en", "english"):
        return "english"
    return "auto"


def extract_voice_json(ref_wav_path: str, prompt_txt: str, tmp_dir: str) -> str:
    """
    根据参考wav+参考文本，生成音色json文件
    qwen3‑tts‑gguf 需要音色json，不能直接喂wav给clone
    返回生成好的音色json路径
    """
    # 创建临时stream做音色提取
    stream = engine.create_stream()
    # set_voice支持wav文件，内部会提取特征生成音色
    stream.set_voice(ref_wav_path, prompt_text=prompt_txt)
    voice_json_path = os.path.join(tmp_dir, f"voice_{os.urandom(8).hex()}.json")
    stream.voice.save(voice_json_path)
    return voice_json_path


def get_voice(ref_audio_path: str, prompt_text: str):
    """
    取出（或首次构建）音色锚点，带进程内缓存。

    为什么需要：构建音色锚点与请求文本无关，但包含三段重活
        1. CodecEncoder.encode()   —— encoder.py 目前硬编码 CPUExecutionProvider
        2. SpeakerEncoder.encode() —— 同上
        3. _normalize 里为拿 final_state 做的一次完整预解码
    实测约 430 ms/次（59-Test-ServerApi-Flow.py）。短文本配音时它会直接吃掉大半 RTF。
    """
    try:
        mtime = os.path.getmtime(ref_audio_path)
    except OSError:
        return None
    key = (os.path.abspath(ref_audio_path), mtime, prompt_text or "")

    hit = VOICE_CACHE.get(key)
    if hit is not None:
        VOICE_CACHE.move_to_end(key)
        return hit

    tmp = engine.create_stream()
    if tmp is None:
        return None
    try:
        res = tmp.set_voice(ref_audio_path, prompt_text)
    finally:
        tmp.shutdown()
    if not res:
        return None

    VOICE_CACHE[key] = res
    while len(VOICE_CACHE) > VOICE_CACHE_MAX:
        VOICE_CACHE.popitem(last=False)
    return res


@app.get("/tts")
async def voice_clone(
    text: str = Query(..., description="待合成文本"),
    prompt_text: str = Query(..., description="参考音频对应的文字"),
    ref_audio_path: str = Query(..., description="服务器本地参考音频绝对/相对路径"),
    text_lang: str = Query("chinese", description="语言 Chinese / English / Auto")
):
    if engine is None:
        return Response(content="模型尚未加载", status_code=503)

    if not os.path.exists(ref_audio_path):
        return Response(content=f"参考音频不存在：{ref_audio_path}", status_code=400)

    if len(text) > MAX_TEXT_CHARS:
        return Response(content=f"文本过长，最大{MAX_TEXT_CHARS}字符", status_code=400)

    lang = convert_lang(text_lang)

    t_req = time.perf_counter()
    try:
        # 1) 音色锚点：走缓存，命中时几乎零成本
        voice = get_voice(ref_audio_path, prompt_text)
        if voice is None:
            return Response(content="音色提取失败（检查参考音频与编码器）", status_code=500)
        t_voice = time.perf_counter()

        # 2) 每次请求仍需一个独立的 stream（它持有 talker/predictor 的 KV 上下文）
        stream = engine.create_stream()
        if stream is None:
            return Response(content="创建语音流失败", status_code=500)
        if not stream.set_voice(voice):
            return Response(content="装载音色锚点失败", status_code=500)
        t_stream = time.perf_counter()

        config = TTSConfig(
            temperature=0.7,
            sub_temperature=0.8,
            seed=42,
            sub_seed=45,
            streaming=False,  # API场景关闭流式，一次性拿完整结果
        )

        # 3) 生成
        result = stream.clone(text, config=config)
        if result is None or result.audio is None:
            return Response(content="合成失败", status_code=500)
        t_gen = time.perf_counter()

        # 读取音频数据到内存buffer
        wav_np, sr = result.audio, result.sample_rate

        buf = io.BytesIO()
        sf.write(buf, wav_np, sr, format="WAV")
        buf.seek(0)

        audio_s = len(wav_np) / sr
        gen_s = t_gen - t_stream
        rtf = gen_s / audio_s if audio_s else 0.0
        total = time.perf_counter() - t_req
        print(f"[TTS] 文本 {len(text)} 字 | 音频 {audio_s:.2f}s | "
              f"音色 {(t_voice - t_req) * 1e3:.0f}ms | 建流 {(t_stream - t_voice) * 1e3:.0f}ms | "
              f"生成 {gen_s:.2f}s | 生成 RTF {rtf:.3f} | 总 {total:.2f}s | "
              f"端到端 RTF {total / audio_s if audio_s else 0:.3f}")

        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers={
                "X-TTS-Audio-Sec": f"{audio_s:.3f}",
                "X-TTS-RTF": f"{rtf:.3f}",
                "X-TTS-Voice-Load-Ms": f"{(t_voice - t_req) * 1e3:.0f}",
            },
        )

    except Exception as e:
        return Response(content=f"合成失败: {str(e)}", status_code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3‑TTS‑GGUF FastAPI服务")
    parser.add_argument("--model-dir", required=True, help="GGUF模型目录路径，传给TTSEngine(model_dir=xxx)")
    parser.add_argument("--port", type=int, default=9880, help="监听端口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--onnx-provider", default="GPU", choices=["GPU", "CUDA", "DML", "CPU"],
                        help="ONNX(编解码器)执行后端，GPU=自动挑选可用加速后端")
    parser.add_argument("--predictor-cpu", action="store_true",
                        help="把 Predictor(工匠模型) 放到 CPU 上跑。"
                             "单路配音建议打开：实测端到端 RTF 2.14 -> 1.15（约 1.9x）。"
                             "原理见 57/58-Probe-*.py —— Predictor 每帧 16 次串行 decode，"
                             "每次在 Vulkan 上约 9.6ms 固定开销，放 CPU 降到约 4.6ms。"
                             "批量(多路 lockstep)场景请勿打开，那时固定开销会被摊薄，GPU 更划算。")

    args = parser.parse_args()
    print(f"正在初始化TTSEngine，model_dir={args.model_dir}, "
          f"onnx_provider={args.onnx_provider}, "
          f"predictor={'CPU' if args.predictor_cpu else 'GPU'}")
    engine = TTSEngine(
        model_dir=args.model_dir,
        onnx_provider=args.onnx_provider,
        predictor_use_gpu=(False if args.predictor_cpu else None),
    )

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port
    )
