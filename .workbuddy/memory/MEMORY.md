# Qwen3-TTS-GGUF 项目长期记忆

## 硬件与运行时（开发机）
- GPU：**AMD Radeon RX 6750 GRE 10GB**（`uma: 0`，独立显存，约 384 GB/s）。**本机没有 NVIDIA 卡**，`nvidia-smi` 不存在
- llama.cpp 后端：`inference/bin/` 只有 **Vulkan**（`ggml-vulkan.dll`），无 CUDA / HIP
- onnxruntime：`onnxruntime-directml==1.24.4`，可用 provider 只有 `DmlExecutionProvider` 和 `CPUExecutionProvider`
  → "GPU" provider 实际落到 **DirectML**；Decoder 跑在子进程 `workers/decoder.py`
- 模型：talker `q5_k` 1006MB / predictor `q8_0` 151MB / decoder+codec_enc+spk_enc 为 fp16 ONNX

## 性能结论（已实测，勿重复调研）
**单路慢的根因 = Predictor 每次 `llama_decode` 在 Vulkan 上约 9.6 ms 固定开销**
- Predictor 每帧固定 16 次串行 decode（1 次 prefill + 15 步自回归），**调用次数与并发路数无关**
- 该开销与 batch 内 token 数几乎无关：1 token = 9.70 ms，2 = 9.84，16 = 12.50，512 = 64.39
  → 16 次「1 token」调用 155.22 ms vs 1 次「16 token」调用 12.50 ms，差 12.4x
- 与 `n_ctx` / `n_batch` / `embeddings` / `flash_attn` **无关**（参数 A/B 全部无效）
- 不是算力、不是显存带宽、不是跨模块拷贝、不是 DML/ONNX 干扰
- 单路 RTF ≈ 9.6 ms × 16 × 12.5 fps ≈ 1.9，被此固定开销钉死

**两条提速路线**
1. **多路批量**（`BatchRunner`，多路 lockstep）：同一批 16 次调用服务 N 路
   → 每路 RTF：B=1 = 1.965，B=16 = 0.208，B=32 = 0.135
2. **单路：Predictor 放 CPU**：每次调用 4.6 ms（vs Vulkan 9.8 ms）→ 端到端 RTF 2.06 → 1.03
   - 开关：`TTSEngine(predictor_use_gpu=False)` / `ServerApi.py --predictor-cpu`
   - **批量场景不要开**（那时固定开销已被摊薄，GPU 更划算）

## 踩坑与约定
- **任何 decode 探针必须断言返回码 `rc == 0`**。llama.cpp 在位置不连续时会拒绝整批
  （`init: ... required that Y = X + 1` → `decode: failed to initialize batch`，ret = -1），
  否则量到的是"失败调用"的耗时，数据全是假的（本项目 `54-Probe-Logits-Latency.py` 就踩过）
- 自回归步进的位置必须**连续递增**：prefill 2 token 占 pos 0/1，后续第 cs 步起始 pos 用 `cs + 1`
- 每次 `llama_decode` 后若立即读输出（`get_logits_ith` / `get_embeddings`），等待就发生在读的那一步；
  若隔了几十毫秒再读，则接近零成本（talker 的 logits 读只要 12.5 µs 就是这个原因）
- `encoder.py` 的 `CodecEncoder` / `SpeakerEncoder` **硬编码 `CPUExecutionProvider`**（第 13/47 行），
  GUI 的 ONNX provider 选择器对它们无效 —— 已知疏漏，未修
- 文档 `CLAUDE.md` / `readme.md` 里 "Encoder/Decoder: ONNX Runtime (DirectML/Cuda)" 的描述与实际一致，
  但没提"编码器走 CPU"这一层

## 诊断脚本
- `53-Profile-Breakdown.py`：全链路插桩归因（`--compare A B` / `--no-mask`）
- `55-Probe-Decode-Scaling.py`：decode + 首次读 logits 随批大小变化
- `56-Probe-Ctx-Params.py`：context 参数 A/B
- `57-Probe-Token-Scaling.py`：单次调用内 token 数扫描（判定固定开销 vs 算力）
- `58-Probe-Predictor-Backend.py`：Vulkan vs CPU 后端对比
- `59-Test-ServerApi-Flow.py`：复刻 ServerApi 单请求流程并拆分固定开销
