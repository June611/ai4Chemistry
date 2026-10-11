# 部署决定

## 2026-10-11：迁移到用户指定服务器

用户授权在/root/autodl-tmp/evan克隆项目、配置uv并SCP迁移数据。
本次仅环境/数据部署及必要验证，不启动正式训练；训练仅针对QM9。
发现直连GitHub clone超时、默认pip镜像无法取uv；按用户建议使用SSH反向代理。
服务器127.0.0.1转发到WSL现有代理，公网不监听代理端口；旧连接中断后改用17891。
uv安装至/root/.local/bin，项目.venv与缓存放数据盘，原Conda环境保留。

## QM9数据边界

活动模型配置必须指向qm9_full的3407/42/2026划分，原始源为qm9_chn.csv。
数据包中的CN/test子集均属QM9历史诊断，不纳入正式入口。
Infomax预训练最多取当前train的50000样本，验证集用于选模，测试集不参与拟合或预训练。
预训练checkpoint需要匹配manifest、geometry和seed；不使用QMugs/QMv4/外部权重。
新增脚本audit_qm9_training_scope.py检查来源及配置路由，完整样本审计由split审计器负责。

## GPU依赖边界

主uv.lock先原样安装。DGL的CPU wheel不能证明GPU可用；GPU部署需要实际设备探测。
准备DGL官方同版本1.1.3+cu121的Python3.11 Linux wheel，来源：
https://data.dgl.ai/wheels/cu121/repo.html 。是否替换及uv是否会重新同步须实测并记录。
不以环境兼容变更改变模型、loss/scaler、LR、256/100/20或三组配对规则。

## 下载策略与当前进度

uv已安装，独立Python3.11.15和.venv已创建；创建环境不等于依赖安装完成。
官方PyPI的测试下载约66KB/s，清华镜像约2.2MB/s。优先从清华镜像直接获取Python包，
GitHub和受限PyTorch索引走反向代理。使用uv.lock导出的requirements和--require-hashes，
不运行uv add、不更改锁定版本；安装后再核对锁文件与实际环境。
DGL GPU wheel已在本地完整下载并SCP至服务器artifacts，约77MB；SHA256：
07e2a27f7d30c6c0f5a04384390298977207f6df0ff832ce09fa644f893af0fa。
其安装和CUDA验证仍待基础依赖就绪，当前无需用户手动下载DGL。

GitHub源码下载中断留下的进程仅停止本次部署所属任务，不触及其他作业。
剩余DimeNet、3DInfomax、LieConv改从本地已核验且无修改的源码仓库SCP迁移，
保留Git信息并在服务器重新核对sources.lock.json的精确commit。

## 依赖安装完成及DGL保护

基础依赖最终通过拆分锁文件导出的requirements完成：Torch单独从官方cu124索引的缓存安装，
其他依赖使用清华镜像。全部保持锁定版本与哈希；uv sync --locked --offline --extra qm9通过。
随后安装DGL1.1.3+cu121；Torch2.6.0+cu124的CUDA矩阵运算和DGL图迁移均在RTX3090通过。
这只证明CUDA基础依赖可用，逐模型检查和batch256显存验收仍需分别记录。
uv sync --locked --offline --extra qm9 --dry-run实测会将GPU DGL替换为CPU1.1.3。
服务器激活文件设置UV_NO_SYNC=1，运行时使用已安装环境；显式uv sync仍会恢复CPU版本，
需再次安装本地GPU wheel。预检脚本增加可选QM9_DGL_GPU_WHEEL，同步后恢复GPU构建并禁用运行时同步。
该选项不改变默认安装路径，也不改变模型训练口径。

## 用户纠正：直接声明GPU构建，替代手动恢复方案

用户指出本地具备GPU，应直接安装GPU版；本地实测GTX1050Ti，原DGL1.1.3为CPU构建。
已在pyproject.toml的uv.sources中声明Linux/x86_64/Python3.11使用官方DGL1.1.3+cu121 wheel，
其他环境保留原有来源；uv.lock据此重新生成。本地CUDA探测通过，后续同步dry-run不再替换DGL。
这一声明替代前一条UV_NO_SYNC、QM9_DGL_GPU_WHEEL方案；预检脚本保持普通uv sync即可。
配置来源使用uv支持的条件来源：[官方说明](https://docs.astral.sh/uv/concepts/projects/dependencies/#platform-specific-sources)。

服务器95项现有测试通过，真实QM9 train样本的batch2 GPU前向/反向9/10通过。
TorchMD-Net失败来自Warp1.18.0默认CUDA13构建，要求R580驱动；当前服务器R570，
CPU测试无法暴露该问题。改用同版本1.18.0+cu12官方构建，不修改模型、邻接算法或训练超参数。
来源：[NVIDIA安装说明](https://nvidia.github.io/warp/v1.18/user_guide/installation.html#cuda-requirements)。
兼容修复后，服务器十个adapter的batch2 GPU前向/反向全部通过，TorchMD-Net受影响CPU测试复验通过。
本地GTX1050Ti的DGL设备检查、TorchMD-Net GPU前向/反向均通过，锁定同步dry-run无变更。
其他非DGL/Warp包的锁定版本保持原样；两者只更换同版本CUDA构建。
三组实际入口在服务器复核：63通过、0失败；3个Infomax下游入口因尚未正式预训练保留待验收。
Infomax GPU通过项仅为随机初始化PNA诊断，不冒充预训练模型或正式实验结果。
