from fastapi import FastAPI, Query
from fastapi.responses import Response
import io
import soundfile as sf
import os
import argparse
import tempfile
from pathlib import Path

# 真实的 qwen3‑tts‑gguf 导入
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig

app = FastAPI(title="Qwen3‑TTS‑GGUF Local API")

engine: TTSEngine | None = None
ROOT = Path(__file__).resolve().parent
MAX_TEXT_CHARS = 1000


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

    # 临时目录存放生成的音色json，请求结束可以删掉
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            #voice_json = extract_voice_json(ref_audio_path, prompt_text, tmpdir)

            stream = engine.create_stream()
            stream.set_voice(ref_audio_path, prompt_text)

            config = TTSConfig(
                temperature=0.8,
                sub_temperature=0.8,
                seed=42,
                sub_seed=45,
                streaming=False,  # API场景关闭流式，一次性拿完整结果
            )

            result = stream.clone(text, config=config)
            stream.join()  # 等待合成完成

            # 读取音频数据到内存buffer
            wav_np, sr = result.audio, result.sample_rate

            buf = io.BytesIO()
            sf.write(buf, wav_np, sr, format="WAV")
            buf.seek(0)
            return Response(content=buf.read(), media_type="audio/wav")

        except Exception as e:
            return Response(content=f"合成失败: {str(e)}", status_code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3‑TTS‑GGUF FastAPI服务")
    parser.add_argument("--model-dir", required=True, help="GGUF模型目录路径，传给TTSEngine(model_dir=xxx)")
    parser.add_argument("--port", type=int, default=9880, help="监听端口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")

    args = parser.parse_args()
    print(f"正在初始化TTSEngine，model_dir={args.model_dir}")
    engine = TTSEngine(model_dir=args.model_dir)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port
    )
