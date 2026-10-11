# QM9 算力服务器部署

日期：2026-10-11。状态：代码、数据、源码、基础CUDA和95项测试完成；统一GPU锁文件的服务器同步及TorchMD-Net复验进行中。
服务器项目路径：`/root/autodl-tmp/evan/ai4Chemistry`；GPU为RTX3090/24GiB。
最初部署提交：`dc2023bd797aeec633a9c9056b5d1c5e29363e8e`。

本功能只部署环境和数据、做必要sanity检查，不启动正式100-epoch训练。
训练范围仅QM9，配对seed3407/42/2026；Infomax只用对应QM9训练集预训练，不用外部权重。
历史QM9 CN/test子集可以随数据包保存，正式配置始终只读取qm9_full三套固定划分。

入口：[tasks.md](tasks.md)、[decisions.md](decisions.md)。
训练规则仍以[统一协议](../paired-seed-100epoch-benchmark/README.md)为准。
密码不写入任何文档、命令文件或Git；部署日志/机器特定配置放artifacts。

## GPU依赖与运行入口

本地与服务器均使用Linux x86_64、Python3.11。该环境的uv.lock直接选取
Torch2.6.0+cu124、DGL1.1.3+cu121、Warp1.18.0+cu12；DGL来自官方data.dgl.ai，
Warp来自NVIDIA官方发布。其他平台/版本按条件来源保留原有解析，不视为已验收GPU环境。
GPU构建固定在pyproject.toml及uv.lock，正常同步不会将DGL换回CPU版，
也不需要UV_NO_SYNC或手动反复安装；使用命令：

```bash
uv sync --locked --extra qm9
uv run --extra qm9 python scripts/check_qm9_device.py --require-cuda
```

服务器非登录shell默认找不到uv；进入项目后加载本次部署的激活文件：

```bash
cd /root/autodl-tmp/evan/ai4Chemistry
source artifacts/server_deployment/server-env.sh
```

DGL、Warp GPU构建与CPU构建属于同一算法版本的环境兼容修复。
batch256/100epochs/patience20、配对seed、loss/scaler、gap MAE选模均保持原协议。
环境预检与小批量GPU sanity不能替代batch256显存或正式三组训练验收。
