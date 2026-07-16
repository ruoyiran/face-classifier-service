# 安全文档

## 敏感数据分类

以下数据按敏感数据处理：

- 原始图片。
- 人脸检测框和关键点。
- aligned crop。
- 人脸质量分。
- embedding 向量。
- Bearer token。
- 加密私钥、AES key、nonce 和 ciphertext。

生产日志不得打印原始图片、aligned crop、embedding vector、私钥、AES key 或 Bearer token。

## 认证

启用方式：

```env
FACE_AUTH_ENABLED=true
FACE_API_TOKENS=token1,token2
FACE_AUTH_DISABLE_DOCS=true
```

认证启用后，`/healthz`、`/readyz`、`/v1/encryption/public-key` 和所有人脸接口都需要 `Authorization: Bearer <token>`。

token 要求：

- 使用足够长的随机值。
- 不要提交到 Git。
- 定期轮换。
- 多 token 轮换期间可用逗号同时配置旧 token 和新 token。

## 加密上传

启用方式：

```env
FACE_ENCRYPTION_ENABLED=true
FACE_REQUIRE_ENCRYPTED_UPLOADS=true
FACE_ENCRYPTION_PRIVATE_KEY_PATH=config/server_private.key
FACE_ENCRYPTION_PUBLIC_KEY_PATH=config/server_public.key
```

加密接口使用 `AES-256-GCM+SealedBox-v1`：

- 图片内容用 AES-256-GCM 加密。
- AES key 用服务 public key 通过 SealedBox 加密。
- 服务响应使用上传 item 的 AES key 加密。

`FACE_REQUIRE_ENCRYPTED_UPLOADS=true` 时，明文 `/v1/faces/*` 接口返回 `403`。

## 密钥管理

- 私钥只部署在服务端。
- public key 可通过 `/v1/encryption/public-key` 分发。
- 私钥文件权限应限制为服务用户可读。
- 轮换密钥时，客户端必须重新拉取 public key。
- 不要把 `config/server_private.key` 提交到 Git。

## 网络安全

服务默认只建议绑定 `127.0.0.1`，由反向代理对外暴露。反向代理负责：

- TLS。
- 内网访问控制或 IP allowlist。
- 请求体大小限制。
- 限流。
- 日志脱敏。

不要在公网裸露 Uvicorn 服务。

## 数据存储责任

本服务不负责持久化人脸结果。调用方如果存储 embedding 或人脸元数据，应自行实现：

- 加密存储。
- 访问控制。
- 数据保留和删除策略。
- 审计日志。
- 用户授权和合规流程。

