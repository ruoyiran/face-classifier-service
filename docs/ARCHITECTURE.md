# 架构文档

## 总体架构

服务是单进程 FastAPI 应用，启动时加载配置、校验模型 artifact，并通过可配置 analyzer factory 创建模型流水线。请求进入后完成图片解码、大小限制、模型推理和响应序列化。

核心运行链路：

```text
HTTP request
  -> auth / body limit / encrypted payload decrypt
  -> decode image + EXIF transpose + RGB conversion
  -> detector
  -> 106-point landmark
  -> 5-point derivation
  -> ArcFace 112x112 similarity alignment
  -> quality score
  -> KPRPE embedding
  -> L2 normalize + serialize
  -> optional encrypted response
```

## 模块职责

- `main.py`：创建默认 `Settings` 和 ASGI `app`。
- `settings.py`：读取 `.env` 和环境变量，规范化 `API_BASE_PATH`，校验批量大小和认证配置。
- `api.py`：定义 HTTP 路由、中间件、认证、加密上传、请求体限制、错误映射和响应序列化。
- `analysis.py`：定义 analyzer 协议、结果 dataclass、默认模型加载流程和未就绪 analyzer。
- `models.py`：定义本地模型 artifact manifest，并校验文件存在、非空和 `.sha256`。
- `preprocessing.py`：定义图片解码、ArcFace 模板、106 点到 5 点派生、无 shear 的 aligned crop、normalization 和 KPRPE 输入。
- `pipeline.py`：把 detector、landmark、quality、embedder 组合成 `FacePipelineAnalyzer`。
- `native_analyzer.py`：当前原生模型运行器，加载 SCRFD、2d106det、CR-FIQA-L 和 CVLFace KPRPE。
- `auth.py`：Bearer token 校验。
- `encryption.py`：`AES-256-GCM+SealedBox-v1` 加解密工具。

## 配置边界

`Settings` 使用 `FACE_` 前缀，同时兼容部分无前缀别名：

- `API_BASE_PATH` / `FACE_API_BASE_PATH`
- `FACE_MODELS_DIR`
- `FACE_ANALYZER_FACTORY`
- `FACE_DEVICE`
- `FACE_MAX_IMAGE_BYTES`
- `FACE_MAX_ENCRYPTED_IMAGE_BYTES`
- `FACE_MAX_IMAGE_PIXELS`
- `FACE_MAX_BATCH_SIZE`
- `FACE_AUTH_ENABLED`
- `FACE_API_TOKENS`
- `FACE_ENCRYPTION_ENABLED`
- `FACE_REQUIRE_ENCRYPTED_UPLOADS`

## 并发模型

API 层使用单个 `asyncio.Semaphore(1)` 包住模型推理，避免多个请求同时争抢 GPU 显存。Uvicorn 生产部署建议 `--workers 1`，除非每个 worker 有独立 GPU 或明确接受重复加载模型带来的显存占用。

## 模型未就绪行为

启动时会先校验模型 artifact。如果文件缺失、SHA 不匹配或 analyzer factory 加载失败，服务仍可启动，但 analyzer 为未就绪状态：

- `/healthz` 返回 `200`。
- `/readyz` 返回 `503`。
- 人脸分析接口返回 `503 model is not ready`。

## 可扩展点

唯一推荐模型扩展点是 `FACE_ANALYZER_FACTORY=module:function`。该函数接收 `Settings`，返回满足 `FaceAnalyzer` 协议的对象。新增实现必须保持公开 API 和响应结构兼容。
