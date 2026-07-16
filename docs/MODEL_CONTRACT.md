# 模型与预处理契约

## 模型清单

V1 固定使用以下模型：

- `detector`: `insightface/scrfd_34g`
- `landmark`: `insightface/2d106det`
- `quality`: `fdbtrs/CR-FIQA-L iresnet100`
- `embedder`: `minchul/cvlface_adaface_vit_base_kprpe_webface12m`

模型 artifact 必须在本地预置，服务启动时只校验，不在实时请求中下载模型。

## Artifact 布局

默认模型目录为 `/models`，可通过 `FACE_MODELS_DIR` 修改。当前 manifest 要求：

```text
insightface/scrfd_34g.onnx
insightface/scrfd_34g.onnx.sha256
insightface/2d106det.onnx
insightface/2d106det.onnx.sha256
fdbtrs/CR-FIQA-L/181952backbone.pth
fdbtrs/CR-FIQA-L/181952backbone.pth.sha256
minchul/cvlface_adaface_vit_base_kprpe_webface12m/model.safetensors
minchul/cvlface_adaface_vit_base_kprpe_webface12m/model.safetensors.sha256
minchul/cvlface_adaface_vit_base_kprpe_webface12m/wrapper.py
minchul/cvlface_adaface_vit_base_kprpe_webface12m/wrapper.py.sha256
minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.yaml
minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.yaml.sha256
minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.pt
minchul/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.pt.sha256
```

每个 `.sha256` 文件第一段必须是对应文件的 SHA256 hex digest。

## 预处理契约

这些参数是 embedding 兼容性的一部分：

- aligned crop 尺寸：`112x112`。
- crop 模板：ArcFace 五点模板。
- 对齐点：从 106 点派生 5 点，最终使用 5 点估计无反射 similarity transform
  （旋转、等比缩放、平移；不引入 shear 或非均匀缩放）。
- 图像格式：质量模型和特征模型输入必须是 RGB。
- 原始 crop dtype：`uint8`，范围 `0-255`。
- 模型输入 dtype：`float32`。
- 模型输入范围：`[-1, 1]`。
- mean/std：`mean=[0.5,0.5,0.5]`，`std=[0.5,0.5,0.5]`。
- 不使用 ImageNet normalization。
- `PREPROCESS_ID`: `arcface_112_similarity_rgb_mean0.5_std0.5`。

## Landmark 规则

`2d106det` 输出 106 点后，服务用固定索引派生 5 点：

- 左眼：索引 `35, 36` 的均值。
- 右眼：索引 `89, 90` 的均值。
- 鼻尖：索引 `86`。
- 左嘴角：索引 `52`。
- 右嘴角：索引 `61`。

106 点可通过 `include_landmarks106=true` 返回，用于调试和诊断，不参与非刚性变形。

## Embedding 规则

- embedder 输入包含 normalized aligned crop 和 aligned crop 坐标系下的 5 点关键点。
- CVLFace KPRPE keypoints 在原生运行器中按 `points / 112.0` 归一到 crop 相对坐标。
- embedder 原始输出必须先 L2 normalize，再返回给调用方。
- API 默认返回 little-endian `float32_base64`。
- JSON 数组格式仅用于调试，不建议生产传输或存储。

## 兼容性规则

以下变化都必须视为 embedding breaking change：

- 修改 crop 尺寸、模板、颜色空间或 normalization。
- 修改 106 点到 5 点派生规则。
- 修改 embedder 模型或权重。
- 修改 embedding 后处理方式。
- 修改 `PREPROCESS_ID` 含义但不改 ID。

发生 breaking change 时，应创建新的 `preprocess_id` 或 `model_id`，并要求外部业务重建向量索引。
