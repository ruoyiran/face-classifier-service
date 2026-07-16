# 测试文档

## 测试目标

测试需要覆盖三类风险：

- API 行为：路由、base path、认证、加密、错误码和序列化。
- 模型前后处理：EXIF、RGB、112x112 ArcFace crop、normalization、L2 normalize。
- 部署风险：模型 artifact、SHA 校验、GPU runtime、反向代理路径和加密上传。

## 本地测试

运行完整测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

运行 lint：

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

## 重点测试文件

- `tests/test_settings.py`：配置解析和 base path 规范化。
- `tests/test_api.py`：公开 API、认证、加密、错误码、embedding 输出。
- `tests/test_preprocessing.py`：图片解码、EXIF、NMS、ArcFace matrix、normalization、KPRPE 输入。
- `tests/test_pipeline.py`：流水线按需执行 detection、alignment、quality、embedding。
- `tests/test_native_analyzer.py`：原生模型运行器路径、ONNX provider、SCRFD/landmark 输出解码。
- `tests/test_models.py`：模型 manifest、`.sha256` sidecar、缺失和 mismatch 报错。
- `tests/test_encryption.py`：加密 payload 和 response envelope。
- `tests/test_auth.py`：Bearer token 校验。
- `tests/test_analysis.py`：默认 analyzer 加载和未就绪行为。

## 变更到测试映射

- 改 API 路由或响应：运行 `pytest tests/test_api.py`。
- 改 base path：运行 `pytest tests/test_settings.py tests/test_api.py`。
- 改预处理或 embedding：运行 `pytest tests/test_preprocessing.py tests/test_pipeline.py tests/test_native_analyzer.py`。
- 改模型 manifest：运行 `pytest tests/test_models.py tests/test_analysis.py`。
- 改认证或加密：运行 `pytest tests/test_auth.py tests/test_encryption.py tests/test_api.py`。

## API 手工测试

使用 [API_TESTING.md](API_TESTING.md) 中的 curl、Python 和 Go 样例测试：

- `/healthz`
- `/readyz`
- `/v1/encryption/public-key`
- 明文 `/v1/faces/*`
- 加密 `/v1/faces/*/encrypted`
- `embedding_format=json` 调试路径

当设置了 `API_BASE_PATH=/face` 时，测试脚本的 `BASE_URL` 应包含前缀：

```powershell
$BASE_URL = "http://127.0.0.1:8000/face"
```

## 部署测试

使用 [DEPLOYMENT_TESTING.md](DEPLOYMENT_TESTING.md) 对真实部署进行验收。重点检查：

- Windows service 或进程是否存活。
- 反向代理路径和直接服务路径是否一致。
- `FACE_REQUIRE_ENCRYPTED_UPLOADS=true` 时明文接口是否返回 `403`。
- 加密接口是否只返回 envelope，不泄露明文 `results`。
- `/readyz` 是否在模型可用时返回 `200`。
- 模型 artifact 是否通过 SHA 校验。

## 模型集成测试建议

真实模型集成测试需要一组受控测试图片。建议覆盖：

- 单人正脸。
- 多人合照。
- 侧脸和轻微遮挡。
- 模糊、低光、强光。
- 儿童和老年人。
- 远距离小脸。
- HEIC/JPEG/PNG 转码图。

断言重点不是固定检测框坐标，而是流程稳定性、无异常、输出字段完整、embedding 已归一化。

