# AGENTS.md

本文件是本仓库的协作约定，适用于所有在 `face-classifier-service` 中工作的自动化代理和开发者。

## Project Context

- 项目是私有化部署、非商用的人脸原子能力服务，面向手机相册业务。
- V1 只提供检测、对齐、质量分、特征向量，不在服务内做聚类、人物命名、人物库管理、去重、活体检测、年龄/性别识别、云同步或前端 UI。
- 服务以 FastAPI 暴露 REST 批量接口，所有人脸接口挂在 `{API_BASE_PATH}/v1/faces/...`。
- 当前 Python 版本约束来自 `pyproject.toml`：`>=3.12,<3.13`。

## Architecture Rules

- 入口：`src/face_classifier_service/main.py` 创建全局 `app`。
- 应用装配：`src/face_classifier_service/api.py` 负责路由、认证、加密上传、请求体限制和序列化。
- 配置：`src/face_classifier_service/settings.py` 负责环境变量解析和校验。
- 模型协议和默认加载：`src/face_classifier_service/analysis.py`、`models.py`、`native_analyzer.py`。
- 预处理契约：`src/face_classifier_service/preprocessing.py` 是唯一可信来源。
- 不要把聚类、向量库、人物簇策略放进服务核心；这些属于外部业务。

## Model And Preprocessing Contract

这些约束会影响 embedding 兼容性，不能在没有版本迁移方案的情况下修改：

- 对齐后 crop 固定为 `112x112`。
- 仿射对齐使用 ArcFace 五点模板。
- 图像进入质量模型和特征模型前必须是 RGB。
- 像素归一化为 `[-1, 1]`，即 `(x / 255.0 - 0.5) / 0.5`。
- `PREPROCESS_ID` 当前为 `arcface_112_rgb_mean0.5_std0.5`。
- embedding 必须 L2 normalize 后返回，默认格式为 `float32_base64`。
- CVLFace KPRPE embedder 需要 aligned crop 坐标系下的 5 点关键点。

## API Rules

- `API_BASE_PATH` 和 `FACE_API_BASE_PATH` 都可配置 base path。
- 空值和 `/` 表示无前缀；非默认值会被规范化为以 `/` 开头且不以 `/` 结尾。
- 明文接口接收 multipart `files`。
- 加密接口接收 `AES-256-GCM+SealedBox-v1` JSON envelope，并返回加密 response envelope。
- 无人脸返回 `200` 和空 `faces`；坏图返回 `400`；超限返回 `413`；模型未就绪返回 `503`。

## Security Rules

- 人脸图片、关键点和 embedding 都按敏感数据处理。
- 默认不要在日志中打印图片内容、aligned crop、embedding vector、AES key、私钥或 Bearer token。
- 生产环境建议启用 `FACE_AUTH_ENABLED=true`。
- 如果启用加密上传，生产环境建议同时启用 `FACE_REQUIRE_ENCRYPTED_UPLOADS=true`。
- `FACE_AUTH_DISABLE_DOCS=true` 时不要绕过 docs 关闭逻辑。

## Development Commands

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

模型运行依赖单独安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

本地启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn face_classifier_service.main:app --host 127.0.0.1 --port 8000 --workers 1
```

## Testing Expectations

- 改路由、配置、认证、加密或序列化时，至少运行 `pytest tests/test_api.py`。
- 改预处理、对齐、embedding 编码时，至少运行 `pytest tests/test_preprocessing.py tests/test_pipeline.py tests/test_native_analyzer.py`。
- 改模型 manifest 或 artifact 校验时，至少运行 `pytest tests/test_models.py tests/test_analysis.py`。
- 提交前优先运行完整 `pytest` 和 `ruff check .`。

## Documentation Expectations

- 需求、架构、部署或接口变化要同步更新 `docs/`。
- 新增环境变量时同步更新 `.env.example`、`docs/DEVELOPMENT.md` 和 `docs/DEPLOYMENT.md`。
- 新增或修改公开 API 时同步更新 `docs/API.md` 和 `docs/API_TESTING.md`。
- 新增模型或修改模型文件路径时同步更新 `docs/MODEL_CONTRACT.md`、`README.md` 和 artifact manifest。
