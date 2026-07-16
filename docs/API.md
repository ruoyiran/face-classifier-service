# API 文档

## Base Path

所有公开接口支持 `API_BASE_PATH` 配置，默认无前缀。

- 空值和 `/` 都表示无前缀。
- 非默认值必须是 URL path prefix，例如 `/face` 或 `/face-api`。
- 规范化后不会以 `/` 结尾。

示例：

```text
API_BASE_PATH=
POST /v1/faces/analyze

API_BASE_PATH=/face
POST /face/v1/faces/analyze
```

## 通用接口

- `GET {base}/healthz`：进程存活检查。
- `GET {base}/readyz`：模型就绪检查。
- `GET {base}/v1/encryption/public-key`：返回加密上传用 public key；未启用加密时返回 `404`。

## 人脸接口

明文接口接收 multipart 表单字段 `files`，支持批量上传：

- `POST {base}/v1/faces/detect`
- `POST {base}/v1/faces/align`
- `POST {base}/v1/faces/quality`
- `POST {base}/v1/faces/embed`
- `POST {base}/v1/faces/analyze`

加密接口接收 JSON payload，并返回加密 envelope：

- `POST {base}/v1/faces/detect/encrypted`
- `POST {base}/v1/faces/align/encrypted`
- `POST {base}/v1/faces/quality/encrypted`
- `POST {base}/v1/faces/embed/encrypted`
- `POST {base}/v1/faces/analyze/encrypted`

## 查询参数

- `include_landmarks106=true`：在 `align`、`quality`、`embed`、`analyze` 中返回 106 点关键点。
- `include_aligned_crop=true`：在 `align`、`analyze` 中返回 raw RGB aligned crop 的 base64。
- `embedding_format=float32_base64|json`：在 `embed`、`analyze` 中控制 embedding 输出格式。

`embedding_format=json` 仅允许在 `FACE_ALLOW_DEBUG_EMBEDDING_JSON=true` 时使用。

## 响应结构

标准响应：

```json
{
  "results": [
    {
      "image_id": "phone.jpg",
      "width": 4032,
      "height": 3024,
      "faces": [
        {
          "bbox_xyxy": [100.0, 120.0, 280.0, 330.0],
          "det_score": 0.98,
          "landmarks5": [[...], [...], [...], [...], [...]],
          "landmarks106": [[...]],
          "alignment": {
            "image_size": [112, 112],
            "matrix": [[...], [...]],
            "landmarks5_crop": [[...], [...], [...], [...], [...]]
          },
          "quality": {
            "model_id": "fdbtrs/CR-FIQA-L iresnet100",
            "score": 0.77,
            "diagnostics": {
              "score_type": "raw_cr_fiqa_qs",
              "preprocess_id": "arcface_112_similarity_rgb_mean0.5_std0.5"
            }
          },
          "embedding": {
            "model_id": "minchul/cvlface_adaface_vit_base_kprpe_webface12m",
            "dim": 512,
            "preprocess_id": "arcface_112_similarity_rgb_mean0.5_std0.5",
            "normalized": true,
            "format": "float32_base64",
            "vector": "..."
          },
          "warnings": []
        }
      ],
      "warnings": []
    }
  ]
}
```

字段会随接口能力裁剪。例如 `detect` 不返回 `alignment`、`quality` 和 `embedding`。

## 错误语义

- `200` + `faces: []`：图片有效但未检测到人脸。
- `400 invalid image`：文件不是可解析图片。
- `400 invalid encrypted payload`：加密 payload 格式错误、密钥不匹配或内容被篡改。
- `401 missing bearer token`：启用认证但没有 Bearer token。
- `401 invalid bearer token`：Bearer token 不在配置列表中。
- `403 encrypted uploads are required`：生产要求加密上传但调用了明文接口。
- `403 json embedding output requires allow_debug_embedding_json=true`：未开启调试向量 JSON 输出。
- `404 encrypted uploads are not enabled`：未启用加密接口。
- `413 image exceeds max_image_bytes`：单张图片超过字节限制。
- `413 image exceeds max_image_pixels`：解码后像素数超过限制。
- `413 batch exceeds max_batch_size`：批量数量超过限制。
- `503 model is not ready`：模型 artifact 或 analyzer 未就绪。

## 加密接口

加密算法固定为 `AES-256-GCM+SealedBox-v1`：

- 客户端生成一次性 AES-256 key 和 AES-GCM nonce 加密图片字节。
- 客户端用服务 public key 通过 SealedBox 加密 AES key。
- 服务解密图片后执行相同分析流程。
- 服务使用第一个上传 item 的 AES key 加密响应，并返回 `key_index: 0`。

加密响应 envelope 只包含：

```json
{
  "alg": "AES-256-GCM+SealedBox-v1",
  "key_index": 0,
  "nonce": "...",
  "ciphertext": "..."
}
```

明文 `results` 只存在于客户端解密后的内容中。

