# 部署 TODO

- [x] D01 检查目标目录，确认空目录、RTX3090/24GiB、50GiB数据盘。
- [x] D02 建立仅绑定服务器127.0.0.1的SSH反向代理，验证GitHub可达。
- [x] D03 浅克隆项目至指定目录；安装uv0.11.7，独立环境不替换Conda。
- [x] D04 SCP迁移归档，校验SHA256、安全解压与86个源文件哈希。
- [x] D05 按uv.lock安装Python3.11.15及qm9依赖；DGL替换为官方1.1.3+cu121，CUDA基础探测通过。
- [x] D06 十个固定官方源码commit均核验一致且无修改；不使用外部预训练权重。
- [x] D07 服务器QM9范围、3407/42/2026三组split/geometry与统一预算审计通过。
- [x] D08 Torch/DGL CUDA通过；十个adapter真实QM9 train batch2 GPU前向/反向通过，95项现有测试通过，Warp兼容修复后TorchMD-Net测试复验通过。
- [x] D09 激活脚本、环境清单及审查TXT已保存；服务器按最终锁文件同步通过，重复同步无变更，普通uv run的CUDA检查通过。
- [ ] D10 batch256单seed一轮已检查：20个下游通过，Cormorant/LieTransformer显存不足；
  完整三组正式训练仍待执行。[烟测证据](../qm9-one-epoch-smoke/README.md)
