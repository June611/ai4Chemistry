# 十模型接入关键决定

## 统一实验协议

十个模型复用 Chemprop 的 `delta_e` Hartree 标签和明确命名的数据划分。当前支持 split seed
3407、42、2026。数据加载必须验证 `split_manifest.csv` 的成员、顺序、seed、分层规则和文件
哈希；模型不能自行重划分。split seed 控制数据成员，training seed 控制初始化、shuffle 和
Dropout，两者作为独立实验维度记录。

当前十模型比较定位为“统一数据与评估管线下的适配实现”。各 adapter 保留论文的关键表示和
消息传递机制，但并非全部复刻原论文的训练时长、容量和优化器。正式结果必须使用
“adapted implementation”表述，不能直接宣称复现了论文数值。

训练前必须冻结一种超参数政策：若采用统一预算协议，所有模型共享数据、标签、选模原则和
最大 epoch，并允许基于显存调整 batch size；模型特有学习率等配置必须在测试前固定并记录来源。
若采用官方配方协议，则逐模型遵循官方优化设置，并把结果单列为 official-recipe track。两种协议
不能混在同一汇总表中。

## 官方源码与兼容补丁

外部仓库固定在 `configs/qm9_models/sources.lock.json`，由 bootstrap 下载到被 Git 忽略的
`.external/qm9_models/`。LieConv、TorchMD-Net 和 3D Infomax 的兼容修改在运行时复制到
`.external/qm9_models/.patched/`，因此官方 checkout 保持可核验且服务器可重复生成补丁。

## 几何数据来源

放弃由 PyG SDF 生成且会静默跳过样本的旧 artifact。正式 artifact 来自 QM9 Figshare
extended-XYZ 归档，下载脚本固定 URL、文件数和 SHA-256。构建过程按原始 `gdb_*` ID 对齐，
重复来源逐个验证，任何未声明的不一致都会失败。

seed 42 和 2026 的几何文件从已验证的 seed 3407 artifact 按对应 train/val/test 顺序无损重排；
每个 artifact 的 JSON 同时记录源几何哈希和目标 manifest 哈希。

官方 `gdb_130667` 文件自身存在已复核的矛盾：首个 SMILES 和 gap 与项目表一致，坐标原子表
却对应同文件的第二个 SMILES/InChI。为保持既有 Chemprop split，只有这个 ID 豁免原子组成
检查；该 ID 会写入几何 JSON 的 `known_official_record_conflicts`。

## 3D Infomax

3D Infomax 是两阶段方法。下游模型是官方 2D PNA，并从 QM9 对比预训练 checkpoint 的
`model_state_dict` 迁移 `node_gnn` 权重。正式 YAML 设置 `require_pretrained: true`；缺少权重时
立即失败。合成 sanity test 可显式关闭此门禁，只用于验证网络接口和梯度。

每个 split 必须单独预训练。checkpoint 同时记录并校验 split manifest 和 geometry SHA-256，
因此 seed 3407 的预训练权重不能用于 seed 42 或 2026。

## 本地验收范围

本地只做结构化 sanity checks 和正式配置 dry-run，不运行完整 epoch。正式的三组 paired
split-seed 训练留给在线 GPU；脚本按模型顺序执行，避免同时占用过多显存。
