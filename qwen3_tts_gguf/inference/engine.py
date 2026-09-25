"""
engine.py - Qwen3-TTS 核心引擎
负责资源管理、模型初始化、音频特征提取及渲染。
"""
import os
import ctypes
from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path
from . import llama, logger
from .assets import AssetsManager
from .stream import TTSStream
from .proxy import DecoderProxy
from .encoder import CodecEncoder, SpeakerEncoder

class TTSEngine:
    """
    Qwen3-TTS 引擎：资源池与 Stream 工厂。
    """
    def __init__(self, model_dir="model", onnx_provider="CUDA", llm_use_gpu=True, chunk_size=12, verbose=True, subprocess_decoder=True, load_llm=True,
                 predictor_use_gpu: Optional[bool] = None):
        """
        Args:
            predictor_use_gpu: Predictor (工匠模型) 是否放 GPU。
                None (默认) = 跟随 llm_use_gpu，保持原有行为。
                False       = Predictor 放 CPU，Talker 仍在 GPU。

        为什么需要这个开关：
            Predictor 每帧要做 16 次串行 llama_decode（1 次 prefill + 15 步自回归），
            而每次调用在 Vulkan 上都有约 9.6 ms 的固定开销（与 batch 内 token 数几乎无关：
            1 token = 9.70 ms，512 token = 64.39 ms）。单路时每帧就是 16 × 9.6 ≈ 155 ms，
            而一帧音频只有 80 ms —— 单路 RTF 因此被钉在 ~1.9。
            实测把 Predictor 挪到 CPU 后每次调用降到 ~4.6 ms，单路端到端 RTF 由 2.14 降到 1.15。
            （见 58-Probe-Predictor-Backend.py / 59-Test-ServerApi-Flow.py）
            批量场景（BatchRunner 多路 lockstep）能把这 9.6 ms 摊到多路上，放 GPU 更合适。
        """
        import time
        import numpy as np
        from tokenizers import Tokenizer
        
        t_start = time.time()
        self.ready = False
        self.model_dir = Path(model_dir)
        self.chunk_size = chunk_size
        
        # 路径定义 (全线使用 Path 对象)
        self.paths = {
            "talker_gguf": self.model_dir / "qwen3_tts_talker.q5_k.gguf",
            "predictor_gguf": self.model_dir / "qwen3_tts_predictor.q8_0.gguf",
            "decoder_onnx": self.model_dir / "qwen3_tts_decoder.fp16.onnx",
            "codec_enc_onnx": self.model_dir / "qwen3_tts_codec_encoder.fp16.onnx",
            "spk_enc_onnx": self.model_dir / "qwen3_tts_speaker_encoder.fp16.onnx",
            "tokenizer": self.model_dir / 'tokenizer.json',
        }
        
        # 核心文件预检（LLM 权重仅推理需要，纯编码/解码工具链可跳过）
        needed = ["decoder_onnx", "tokenizer"]
        if load_llm:
            needed += ["talker_gguf", "predictor_gguf"]
        missing = [name for name, p in self.paths.items()
                  if name in needed and not p.exists()]
        
        if missing:
            logger.error(f"❌ 引擎初始化失败: 缺少核心模型文件 {missing}")
            return

        try:
            # 1. 资产加载
            t_assets = time.time()
            self.assets = AssetsManager(str(self.model_dir))
            self.tokenizer = Tokenizer.from_file(str(self.paths['tokenizer']))
            if verbose: print(f"📦 [Engine] 资产与词表加载完成 (耗时: {time.time()-t_assets:.2f}s)")
            
            # 2. 音频及说话人编码器 (CPU 轻量型)
            self.codec_encoder = None
            self.speaker_encoder = None
            if self.paths["codec_enc_onnx"].exists() and self.paths["spk_enc_onnx"].exists():
                t_enc = time.time()
                self.codec_encoder = CodecEncoder(str(self.paths["codec_enc_onnx"]))
                self.speaker_encoder = SpeakerEncoder(str(self.paths["spk_enc_onnx"]))
                if verbose: print(f"🎤 [Engine] 编码器加载完成 (耗时: {time.time()-t_enc:.2f}s)")

            # 3. 解码后端: 子进程 (流式播放场景) 或进程内 (GUI/离线批量场景)
            t_parallel = time.time()
            if subprocess_decoder:
                self.decoder = DecoderProxy(str(self.paths["decoder_onnx"]), onnx_provider=onnx_provider, chunk_size=self.chunk_size)
                if verbose: print("⏳ [Engine] 正在拉起子进程解码器...")
            else:
                from .decoder import LocalDecoder
                self.decoder = LocalDecoder(str(self.paths["decoder_onnx"]), onnx_provider=onnx_provider, chunk_size=self.chunk_size)

            # 4. 模型引擎初始化 (并行点 2: GGUF 在主进程加载，Decoder 在子进程同时初始化)
            if load_llm:
                t_gguf = time.time()
                self._init_llama_engines(llm_use_gpu, predictor_use_gpu)
                if verbose: print(f"🧠 [Engine] GGUF 推理后端就绪 (耗时: {time.time()-t_gguf:.2f}s)")
            else:
                self.talker_model = None
                self.predictor_model = None

            # 5. 子进程模式同步等待解码器信号；进程内模式天然就绪
            if subprocess_decoder:
                is_decoder_ready = self.decoder.wait_until_ready(timeout=10)
                if not is_decoder_ready:
                    print(f"❌ [Engine] 引擎初始化未完全就绪 (解码器超时)。")
                    logger.warning("⚠️ [Engine] 解码器就绪超时，渲染功能将不可用。")
                    self.ready = False
                    return
            self.ready = True
            if verbose:
                mode = "子进程" if subprocess_decoder else "进程内"
                print(f"✅ [Engine] 解码器就绪 ({mode}) (总并行初始化耗时: {time.time()-t_parallel:.2f}s)")

            print(f"🚀 [Engine] 引擎全链路初始化完成! 总耗时: {time.time()-t_start:.2f}s")

        except Exception as e:
            logger.error(f"❌ 引擎初始化过程中出现致命异常: {e}", exc_info=True)
            self.shutdown()

    def __bool__(self):
        return self.ready

    def _init_llama_engines(self, llm_use_gpu, predictor_use_gpu: Optional[bool] = None):
        """初始化 GGUF 模型（仅加载模型，不创建 Context）

        Talker 与 Predictor 可以分置不同设备：Talker 每帧只 1 次 decode，放 GPU；
        Predictor 每帧 16 次串行 decode，单路场景放 CPU 反而更快（详见 __init__ 说明）。
        """
        logger.info("[Engine] 正在加载 GGUF 模型...")

        if predictor_use_gpu is None:
            predictor_use_gpu = bool(llm_use_gpu)

        try:
            # 使用新的 LlamaModel 类
            self.talker_model = llama.LlamaModel(self.paths["talker_gguf"], n_gpu_layers=-1, use_gpu=llm_use_gpu)
            self.predictor_model = llama.LlamaModel(
                self.paths["predictor_gguf"],
                n_gpu_layers=(-1 if predictor_use_gpu else 0),
                use_gpu=predictor_use_gpu,
            )

            logger.info(f"✅ [Engine] GGUF 模型加载完成 "
                        f"(talker={'GPU' if llm_use_gpu else 'CPU'}, "
                        f"predictor={'GPU' if predictor_use_gpu else 'CPU'})。")
        except Exception as e:
            logger.error(f"❌ 加载 GGUF 模型失败 (可能是显存不足/OOM): {e}")
            raise

    def create_stream(self, n_ctx=2048, voice_path: Optional[str] = None) -> Optional[TTSStream]:
        """工厂方法：创建语音流"""
        if not self.ready:
            logger.error("❌ 引擎未就绪，无法创建语音流。")
            return None
        return TTSStream(self, n_ctx=n_ctx, voice_path=voice_path)

    def encode(self, input) -> Optional[np.ndarray]:
        """快捷入口：提取音色特征 (Speaker Embedding)。支持传入 numpy 数组或 TTSResult。"""
        if self.speaker_encoder is None:
            logger.error("❌ 本模型无 SpeakerEncoder")
            return None
        return self.speaker_encoder.encode(input)

    def decode(self, codes, **kwargs) -> np.ndarray:
        """快捷入口：解码渲染音频。支持传入 numpy 数组 (codes) 或 TTSResult。"""
        return self.decoder.decode(codes, **kwargs)



    def shutdown(self):
        """释放资源，支持重新开启引擎"""
        if not hasattr(self, '_already_shutdown'):
            logger.info("[Engine] 正在关闭引擎 (清理显存与子进程)...")
            try:
                if hasattr(self, "decoder"):
                    self.decoder.shutdown()
                # 显式解除引用，触发 __del__ 释放资源
                self.talker_model = None
                self.predictor_model = None
            except Exception as e:
                logger.warning(f"⚠️ 关闭引擎时出现小异常 (忽略): {e}")
            self.ready = False
            self._already_shutdown = True
            logger.info("✅ [Engine] 引擎资源已彻底释放。")

    def __del__(self):
        self.shutdown()
