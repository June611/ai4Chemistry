# 飞书多维表格训练监测

目标：在实际执行训练的 WSL/Linux 主机采样内存、显存和运行耗时，读取本地保存的
测试指标，让每个模型的每次 seed 运行在同一张飞书多维表格中占一行，便于远程查看和回顾。

状态：2026-10-12已接入本地官方Lark CLI同步桥，并通过服务器8项真实诊断与云端回读。
训练表60个字段、硬件采样表8个字段；8条completed记录、7份完整预测附件及完整精度JSON
均已核验。本地全套141项、服务器相关32项测试通过；正式100epoch训练尚未启动。
当前推荐[一键调度入口](../qm9-resource-aware-training/operations.md)；
详见[最终审查](../qm9-resource-aware-training/final_review.md)和[tasks.md](tasks.md)。

## 推荐入口：服务器训练，本地CLI同步

```bash
bash scripts/train_all_auto.sh --remote --directory artifacts/auto_training/formal_20261012
```

服务器运行训练和资源采样；WSL通过SSH读取摘要、预测文件，使用已授权CLI用户登录同步飞书。
飞书登录凭据不复制到服务器；本地目标ID配置`configs/feishu_training.json`不提交Git，模板为
`configs/feishu_training.example.json`。入口先校验表配置和训练表schema再启动远端任务。
关闭同步端不停止服务器训练；恢复同步使用：

```bash
bash scripts/train_all_auto.sh --sync --remote --directory artifacts/auto_training/formal_20261012
```

- [实际训练记录](https://my.feishu.cn/base/Nckeb46n6anmx9shVAXcnul6nsh?table=tblgtkzmgVWdVrWu)
- [实际硬件采样](https://my.feishu.cn/base/Nckeb46n6anmx9shVAXcnul6nsh?table=tblADD5ee9TfNKbM)
- [云端操作说明](https://my.feishu.cn/docx/SLSedXUDao610nxZAH4cqg5onYA)

数字指标最多显示4位小数，“指标完整精度_JSON”、本地JSON与预测CSV保留完整数值。
GPU利用率是整卡utilization.gpu，非SM occupancy；FLOPs表示计算量，本次未测项留空。
推理速度使用实测样本/秒与摊销时间，并记录计时范围。共享运行的整卡利用率不能归因于单模型。
下面保留独立Python应用身份客户端的可选使用说明；一键CLI方式不要求额外配置App Secret。

## 可选：独立Python应用身份客户端

本机已确认是 WSL2。2026-10-12 对 `open.feishu.cn` 的真实公开 HTTPS 请求返回 200，
证明可直接访问官方域名。工具使用 Python 标准库 HTTPS 请求，使用已有 PyYAML 读取配置，
无需安装 Lark SDK，也不需要 Windows 程序或回调服务器。

```text
实际训练主机（WSL 或 GPU 服务器）
  ├─ 训练：config.yaml、metrics.json、training_timing.json、预测 CSV
  └─ 监测：/proc 进程树 + nvidia-smi + 训练历史
       ├─ 本地 tracking/samples.jsonl、state.json、summary.json
       └─ 后台 HTTPS → 国内飞书 API → 同一张多维表格
```

如果模型在远端 GPU 服务器训练，监测工具也要在该服务器运行；WSL 上的 SSH 进程
内存不是远端模型内存。服务器能直接出网时，在服务器配置飞书凭证并执行相同命令。
如果服务器不能出网，使用 `--local-only`，将完整运行目录复制回 WSL 后再 `sync`。
包含 tracking/state.json 的复制会保留监测生成的运行 ID；未监测的历史结果首次回填
按执行回填的主机和绝对目录产生 ID，应固定回填位置以保持行身份。

## Python客户端首次配置

WSL 现有 CLI 已验证可用时，可直接复用其登录。以下步骤适用于单独启用当前 Python
应用身份客户端；不能因为项目环境变量为空而认定 CLI 未配置。

1. 在 [国内飞书开发者后台](https://open.feishu.cn/app) 创建企业自建应用，
   名称可用“模型训练记录”。保存 App ID；App Secret 只保存在运行主机。
2. 在应用权限管理中开通“查看、评论、编辑和管理多维表格” `bitable:app`。
   本工具需要列出/新增字段、查询/新增/更新记录。发布应用版本，并完成租户所需审核。
   如果租户改用细分权限，按这些操作对应的官方接口权限配置。
3. 用自己的飞书账号新建一个独立多维表格，名称可用“QM9 模型实验”，
   将一个数据表命名为“训练运行”。不需要手工创建几十个字段。
4. 在该多维表格的 `··· → 更多 → 添加文档应用` 中加入自建应用并授予编辑权限。
   API scope 与该表格的文档权限是两层授权，开通 scope 后仍需执行这一步。
   [飞书官方配置说明](https://www.feishu.cn/content/137710114294)
5. 复制完整的 Base URL，格式为
   `https://<租户>.feishu.cn/base/<app_token>?table=<table_id>`。
   `/base/` 后面的 token 属于多维表格，不是开发者应用 App ID 或登录 token。
   `/wiki/` 链接里的节点 token 不能直接代替 app_token；初次接入建议新建独立 Base。

在训练主机的项目根目录执行：

```bash
cp configs/feishu.env.example .env.feishu
chmod 600 .env.feishu
# 用本机编辑器填写 .env.feishu：App ID、App Secret、完整 Base URL
source .env.feishu

# 添加缺失字段；仅操作你在环境变量中指定的已有数据表
uv run --locked python -m molgap.tracking init-table

# 只读校验：认证、目标表可访问、字段名称及类型匹配
uv run --locked python -m molgap.tracking check
```

`.env.feishu` 已加入 Git 忽略规则。工具不会记录 App Secret 或 access token，
tenant_access_token 仅保存在进程内并在到期前刷新。
[官方 token 接口](https://open.feishu.cn/document/server-docs/authentication-management/access-token/tenant_access_token_internal)

如果只需查看字段定义，执行以下命令即可，无凭证和网络要求：

```bash
uv run --locked python -m molgap.tracking schema
```

`init-table` 保留已有字段并新增缺少的字段，发现同名字段类型冲突时停止。
当前schema包含59个文本和数字字段，时间用 UTC ISO 文本保存；CLI训练表另加一个预测附件字段。
初始化后可隐藏飞书自带的默认空列。早期43字段验收记录保留为历史，不表示当前schema。

## 已有训练：附加监测

在实际训练主机上，用训练 Python 进程的 PID，而不是 SSH 或整组顺序调度脚本的 PID。
PID 下的 DataLoader、Chemprop 子进程也会计入。

```bash
source .env.feishu
uv run --locked python -m molgap.tracking watch \
  --run-dir outputs/paired_gap_mae_v2/schnet_qm9/seed3407 \
  --pid <训练进程PID> \
  --interval 5 --sync-interval 60
```

尚未配置账号也可开始本地采样：在该命令末尾加 `--local-only`。
附加监测只能获得开始监测以后的耗时和采样峰值；已有 `training_timing.json`
中的 fit 时间会单独读取。`watch` 的 Ctrl+C 停止监测，训练进程继续运行。
重启采样器附加同一个 PID 时，保留累计统计并把覆盖标记为 resumed_partial_sampled。
每个同时运行的模型使用独立 `watch` 进程；同一目录不允许两个采样器同时写入。

## 新训练：从启动开始监测

`run` 在单独进程组启动你指定的训练命令，监测整个训练子进程树并保留训练退出码。
`--run-dir` 必须与该训练命令实际输出目录一致，而且不能已存在。
工具不会提前创建目录，兼容传统 ML 对输出目录必须全新的要求。

```bash
source .env.feishu
uv run --locked python -m molgap.tracking run \
  --run-dir outputs/paired_gap_mae_v2/schnet_qm9/seed3407 \
  --interval 5 --sync-interval 60 \
  -- python -m molgap.qm9_models.train \
     --config configs/qm9_models/schnet.yaml
```

上例仅是操作说明，本次接入不会启动正式模型训练。训练入口仍执行其自身的协议、
暂停模型、seed 和预算校验。`run` 的 Ctrl+C 会停止自己启动的训练进程组并记录 interrupted。
如果要在终端关闭后继续运行，放入已有的 tmux 会话。训练原有控制台日志由训练命令管理。

## 历史回填和断网补同步

```bash
# 只打印摘要，查看将要上传的所有行；不写飞书、不修改本地产物
uv run --locked python -m molgap.tracking sync \
  outputs/paired_gap_mae_v2 --dry-run

# 账号配置完成后执行真实同步
source .env.feishu
uv run --locked python -m molgap.tracking sync outputs/paired_gap_mae_v2
```

可以指定一个模型目录或单次 seed 目录。正式输出和历史诊断输出建议分别回填、用视图
按协议和诊断标记筛选。这里只负责记录，不代替项目的可比性与预测身份审查。

相同运行 ID 查询到已有行就更新；第一次创建使用落盘的 UUID 幂等键，丢失响应时重试
仍使用同一个键。每次同步保存 record_id；发现重复运行 ID 时停止并报告。
接口与方法已对照 [官方 SDK 的新增请求定义](https://github.com/larksuite/oapi-sdk-python/blob/v2_main/lark_oapi/api/bitable/v1/model/create_app_table_record_request.py)、
[查询请求定义](https://github.com/larksuite/oapi-sdk-python/blob/v2_main/lark_oapi/api/bitable/v1/model/search_app_table_record_request.py)
和 [更新请求定义](https://github.com/larksuite/oapi-sdk-python/blob/v2_main/lark_oapi/api/bitable/v1/model/update_app_table_record_request.py) 核对。

网络请求在后台线程执行，采样和训练不等待飞书响应。失败写入 `sync_status.json`，
下一次同步周期重试最新摘要；训练结束仍未同步成功时，用上面的 sync 命令补传。
每次资源样本只写本地 JSONL，不逐条新增飞书行。

## 表格字段与本地文件

| 类别 | 内容 |
|---|---|
| 运行身份 | 模型、训练/划分 seed、协议、运行 ID、主机、运行目录、诊断标记 |
| 状态和配置 | running/completed/failed 等、起止时间、batch、epoch 预算、学习率、loss、标签缩放 |
| 训练过程 | 当前 epoch、训练 loss、最近验证 gap MAE、采样覆盖和数量 |
| 资源 | 当前/采样峰值 RSS、PSS、训练进程显存、整卡显存，全部 MiB |
| 耗时 | 从启动/附加开始的流水线秒数、原训练保存的 fit 秒数、已完成 epoch |
| 最终测试 | delta_e 的 MSE、MAE、RMSE、MAPE、R²、样本数 |

测试 MAE/RMSE 和验证 MAE 使用原始 Hartree，MSE 为 Hartree²，MAPE 为百分比。
训练 loss 保留各模型原始定义，不能跨模型直接当成物理单位测试 MSE 比较。
新训练的回归指标原文件增加 `mse`；旧指标缺 MSE 时仅按已保存 RMSE²补充到摘要，
并在摘要 `mse_source` 中注明，不重写旧指标文件。

SPE/SMILES Transformer 在原有 TensorBoard 之外增加 CSV logger，以便实时读取
epoch/loss/验证指标。QM9 复用 history.csv，其他 Lightning 流程读取原有 metrics.csv。
没有这些文件时，对应过程字段留空，资源和最终指标仍可记录。

```text
outputs/<model>/seed<seed>/
  config.yaml                   # 训练原有配置
  metrics.json                  # 训练原有测试指标，新增运行包括 MSE
  training_timing.json          # 训练原有 fit 耗时
  predictions/test_predictions.csv
  tracking/
    samples.jsonl               # 完整的已记录资源采样序列
    state.json                  # PID 身份、资源峰值、覆盖、退出状态
    summary.json                # 每次运行的本地统一摘要和 MSE 来源
    sync_status.json            # 最近一次同步结果；失败为 pending
    feishu_sync.json            # 目标表身份、创建幂等键、record_id 和成功时间
```

RSS 求和会重复计算共享页，PSS 按共享份额分摊；无法读取 smaps_rollup 时只回退 RSS，
PSS 留空。本地样本记录读取成功的进程数量，便于检查权限导致的覆盖不足。
整卡显存可能包括其他训练或桌面进程，单独存储，不能代替该模型进程显存。
WSL 的 nvidia-smi 可能不支持进程级显存，缺失值留空并注明 unavailable。
资源峰值是离散采样近似值，短于采样间隔的峰值可能漏掉。

早期本地会话的 nvidia-smi 被操作系统阻止，不声称测得WSL真实显存；后续已在RTX3090服务器
完成真实GPU采样和8项诊断验收。服务器证据与本地环境限制分别记录。
已有结果如果当时没有监测，无法事后恢复真实历史内存峰值。

推荐在飞书建立“运行中”“已完成”“失败”“诊断运行”四个筛选视图。
“已完成”的正式比较视图还应筛选对应 protocol_id 与诊断运行=no，
按模型、seed 分组，按测试 MAE 排序。每个 seed 保留独立行，三 seed 汇总仍使用项目现有工具。
