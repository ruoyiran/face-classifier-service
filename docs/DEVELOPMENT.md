# 开发文档

## 环境准备

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

模型运行依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

项目要求 Python `>=3.12,<3.13`。

## 本地运行

复制 `.env.example` 为 `.env`，按需修改：

```powershell
Copy-Item .env.example .env
```

本地开发启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn face_classifier_service.main:app --host 127.0.0.1 --port 8000 --workers 1
```

如果没有本地模型文件或未配置 `FACE_ANALYZER_FACTORY`，服务可以启动，但 `/readyz` 和人脸接口会返回 `503`。

## 常用命令

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest tests\test_api.py
.\.venv\Scripts\python.exe -m ruff check .
```

## 代码结构

```text
src/face_classifier_service/
  main.py              ASGI app 入口
  api.py               FastAPI 路由、认证、加密、序列化
  settings.py          环境变量和配置校验
  analysis.py          analyzer 协议和结果类型
  pipeline.py          模型流水线编排
  preprocessing.py     图片解码、对齐、归一化
  native_analyzer.py   原生模型运行器
  models.py            模型 artifact manifest 和 SHA 校验
  auth.py              Bearer token
  encryption.py        加密上传和加密响应
  embedding.py         embedding base64 和 L2 normalize
```

## 配置说明

主要环境变量：

- `API_BASE_PATH` / `FACE_API_BASE_PATH`：API 前缀。
- `FACE_MODELS_DIR`：模型目录。
- `FACE_ANALYZER_FACTORY`：模型 analyzer factory，例如 `face_classifier_service.native_analyzer:create_analyzer`。
- `FACE_DEVICE`：`cuda`、`cuda:0` 或 `cpu`。
- `FACE_MAX_IMAGE_BYTES`：单张明文图片字节限制。
- `FACE_MAX_ENCRYPTED_IMAGE_BYTES`：单张加密图片 envelope 字节限制。
- `FACE_MAX_IMAGE_PIXELS`：解码后像素限制。
- `FACE_MAX_BATCH_SIZE`：批量数量限制。
- `FACE_ALLOW_DEBUG_EMBEDDING_JSON`：是否允许 JSON embedding 调试输出。
- `FACE_AUTH_ENABLED`：是否启用 Bearer token。
- `FACE_API_TOKENS`：逗号分隔 token 列表。
- `FACE_AUTH_DISABLE_DOCS`：启用认证时是否关闭 docs。
- `FACE_ENCRYPTION_ENABLED`：是否启用加密接口。
- `FACE_REQUIRE_ENCRYPTED_UPLOADS`：是否拒绝明文人脸接口。

## 变更规范

- 新增公开 API：先写 API 测试，再实现路由，并更新 `docs/API.md`。
- 修改预处理：先写或更新 `tests/test_preprocessing.py`，确认是否需要新的 `PREPROCESS_ID`。
- 修改模型 artifact：更新 `models.py` manifest、`README.md` 和 `docs/MODEL_CONTRACT.md`。
- 修改认证或加密：更新 `tests/test_auth.py`、`tests/test_encryption.py`、`tests/test_api.py` 和 `docs/SECURITY.md`。
- 不要把相册聚类、人物库或业务阈值写进服务核心。

