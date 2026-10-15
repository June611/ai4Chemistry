# 执行记录

2026-10-15：建立功能TODO，开始阅读模型来源、统一训练入口和数据契约。使用本地PDF正文核对DeMol；官方仓库用于锁定结构与许可证。未启动训练。

本地完整拓扑准备通过：133798个固定样本，0删除；逐分子校验SDF原子/坐标并验证三split几何只改变顺序。1915个部分sanitization记录保留在忽略的缓存JSON；缓存21.11MiB。36项针对性测试通过；新增官方collator对照和三seed/烟测门禁/负载分配检查。初始功能提交bbc115c，随后追加实测耗时分配及运行交接。

服务器六项dry-run通过：两模型×三个split，参数量176825914/47797473；batch256、epochs100、patience20与原数据行数/hash一致。初次烟测已启动，Transformer-M先完成。补充发现并复现全零坐标sentinel导致单重原子batch依赖，修复后37项通过，将仅重做受影响的Transformer-M GPU烟测；DeMol实现不变。

正式启动验收：两项完整batch256烟测通过；按314.32/43.98秒每epoch分配六组任务。GPU0先DeMol3407再2026；GPU1先DeMol42，再Transformer-M3407/42/2026。dispatcher259387，两卡各有running任务且已进入实际训练批次；粗估总墙钟17.5小时，正式训练尚未完成。烟测CSV、分配CSV和完整启动快照已保存WSL，不提交Git。
