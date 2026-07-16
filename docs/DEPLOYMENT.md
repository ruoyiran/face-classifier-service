# 部署文档

## 部署目标

服务面向私有 GPU 环境部署。推荐 NVIDIA GPU，12GB+ 显存，单 worker 运行。生产环境应部署在内网，通过反向代理提供 TLS、认证、限流和访问日志脱敏。

## 部署包内容

部署目录建议包含：

```text
face-classifier-service/
  .venv/
  .env
  src/
  models/
  config/
    server_private.key
    server_public.key
  logs/
```

模型目录可放在部署目录内，也可用 `FACE_MODELS_DIR` 指向独立磁盘路径。

## Python 依赖

基础依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

GPU 模型依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

`requirements-models.txt` 使用 PyTorch CUDA 12.1 wheel index。

## 模型准备

生产环境必须预置模型 artifact 和 `.sha256` sidecar。校验命令：

```powershell
.\.venv\Scripts\python.exe -c "from face_classifier_service.settings import Settings; from face_classifier_service.models import validate_model_artifacts; s=Settings(); validate_model_artifacts(s.models_dir); print('model_artifacts=valid')"
```

推荐同时保存 `MODEL_SOURCES.json`，记录：

- 模型名称和版本。
- 下载来源。
- 权重文件 SHA256。
- 许可证。
- 训练数据使用限制。
- 下载日期和操作人。

## 推荐 .env

内网生产示例：

```env
API_BASE_PATH=/face
FACE_MODELS_DIR=C:\Users\Admin\deploy\face-classifier-service\models
FACE_ANALYZER_FACTORY=face_classifier_service.native_analyzer:create_analyzer
FACE_DEVICE=cuda
FACE_HOST=127.0.0.1
FACE_PORT=8000
FACE_MAX_IMAGE_BYTES=31457280
FACE_MAX_ENCRYPTED_IMAGE_BYTES=44040192
FACE_MAX_IMAGE_PIXELS=16000000
FACE_MAX_BATCH_SIZE=8
FACE_ALLOW_DEBUG_EMBEDDING_JSON=false
FACE_AUTH_ENABLED=true
FACE_API_TOKENS=<replace-with-long-random-token>
FACE_AUTH_DISABLE_DOCS=true
FACE_ENCRYPTION_ENABLED=true
FACE_REQUIRE_ENCRYPTED_UPLOADS=true
FACE_ENCRYPTION_PRIVATE_KEY_PATH=config/server_private.key
FACE_ENCRYPTION_PUBLIC_KEY_PATH=config/server_public.key
```

## 生成加密密钥

使用项目内工具函数生成 key pair：

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from face_classifier_service.encryption import write_key_pair; write_key_pair(Path('config/server_private.key'), Path('config/server_public.key')); print('keys=created')"
```

私钥文件只允许服务进程读取，不要提交到 Git。

## 启动服务

直接启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn face_classifier_service.main:app --host 127.0.0.1 --port 8000 --workers 1
```

只建议绑定 `127.0.0.1`，由 nginx、IIS、Caddy 或内网网关转发外部请求。

## 健康检查

```powershell
curl.exe -H "Authorization: Bearer <token>" http://127.0.0.1:8000/face/healthz
curl.exe -H "Authorization: Bearer <token>" http://127.0.0.1:8000/face/readyz
```

- `/healthz=200` 表示进程存活。
- `/readyz=200` 表示模型已就绪。
- `/readyz=503` 表示模型 artifact、SHA 或 analyzer factory 存在问题。

## 反向代理要求

反向代理必须保留 path prefix。例如 `API_BASE_PATH=/face` 时，服务看到的请求路径也应是 `/face/v1/faces/analyze`。

生产代理建议提供：

- TLS。
- 请求体大小限制。
- IP allowlist 或内网访问控制。
- 速率限制。
- 访问日志脱敏。
- 超时配置，避免大批量请求长期占用连接。

## 运维检查

上线前检查：

- `pytest` 和 `ruff check .` 通过。
- 模型 artifact 校验通过。
- `/healthz` 和 `/readyz` 返回预期状态。
- 启用认证后，无 token 和错误 token 返回 `401`。
- 要求加密上传后，明文人脸接口返回 `403`。
- 加密人脸接口返回 encrypted envelope。
- OpenAPI docs 在生产认证模式下按预期关闭。

