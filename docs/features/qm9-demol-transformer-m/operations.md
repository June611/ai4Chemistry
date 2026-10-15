# 运行交接

服务器：双RTX4090，项目`/root/autodl-tmp/evan/ai4Chemistry`。首次只读查询两卡均空闲，数据盘约31GiB可用。现有环境已包含项目依赖，正在进一步核对。

本地已完成模型适配、协议配置及36项针对性测试；SDF拓扑缓存准备中。尚未同步本功能、未启动烟测或正式训练。后续追加真实路径、进程和结果；不要把本段当作已启动凭据。

为保留服务器旧工作区的未提交修改，本功能采用独立代码目录，计划为`/root/autodl-tmp/evan/ai4Chemistry_demol_transformer_m_20261015`。共享原项目data与.venv，运行时将新目录src加入PYTHONPATH；Git元数据来自本功能提交，不冒充旧目录的commit。
