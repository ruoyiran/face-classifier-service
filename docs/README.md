# 文档索引

本目录保存 `face-classifier-service` 的项目文档。当前服务是私有化、非商用的人脸原子能力服务，面向手机相册业务调用方。

## 核心文档

- [产品文档](PRODUCT.md)：产品目标、范围、用户、能力边界和验收口径。
- [架构文档](ARCHITECTURE.md)：系统模块、数据流、配置、模型加载和运行时边界。
- [API 文档](API.md)：REST 接口、base path、请求/响应、错误码和加密接口约定。
- [模型与预处理契约](MODEL_CONTRACT.md)：模型清单、artifact 布局、112x112 ArcFace crop、RGB、归一化和 embedding 兼容性规则。
- [开发文档](DEVELOPMENT.md)：本地环境、常用命令、代码结构、变更规范。
- [测试文档](TESTING.md)：单元测试、接口测试、部署测试和模型集成测试策略。
- [部署文档](DEPLOYMENT.md)：GPU 私有化部署、环境变量、模型文件、安全配置和运维检查。
- [安全文档](SECURITY.md)：认证、加密上传、敏感数据处理和日志红线。

## 已有测试指南

- [API_TESTING.md](API_TESTING.md)：curl、Python、Go 调用样例，覆盖明文和加密 API。
- [DEPLOYMENT_TESTING.md](DEPLOYMENT_TESTING.md)：面向本机部署目录和 nginx 前缀的部署验收脚本。

## 维护规则

- 改公开接口时，同步更新 `API.md` 和 `API_TESTING.md`。
- 改模型路径、预处理或 embedding 输出时，同步更新 `MODEL_CONTRACT.md`。
- 改环境变量或部署方式时，同步更新 `DEVELOPMENT.md`、`DEPLOYMENT.md` 和 `.env.example`。
- 改安全默认值时，同步更新 `SECURITY.md`。

