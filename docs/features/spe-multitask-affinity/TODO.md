# SPE 多任务亲和力融合 TODO

更新时间：2026-10-10

## 数据与协议

- [x] D01 同时读取 homo、lumo、delta_e，复用既有 split manifest。
- [x] D02 三个标签的 scaler 均只从 train 拟合，并在物理空间逆变换。
- [x] D03 SPE merges 和 vocabulary 仍只从 train 拟合。

## 模型与损失

- [x] M01 共享 SPE Transformer encoder，建立三个任务表示与回归 head。
- [x] M02 实现逐行 softmax 的可学习 3×3 任务亲和力矩阵。
- [x] M03 实现 delta_e = lumo - homo 的物理一致性损失。
- [x] M04 支持 affinity/consistency 独立开关，形成完整 2×2 消融。

## 训练与产物

- [x] T01 接入统一 run directory、checkpoint、early stopping 和 Hartree 指标。
- [x] T02 输出三个直接预测任务、轨道能级派生 gap 和一致性残差指标。
- [x] T03 输出最终亲和力矩阵、tokenizer、scaler、split 哈希和测试预测。
- [x] T04 接入现有五 training-seed 执行器。
- [x] T05 提供三套 split 可生成的配置和服务器命令。

## 验证

- [x] V01 模型形状、亲和力归一化、开关隔离、梯度和物理残差测试。
- [x] V02 train-only scaler、数据身份和 portable checkpoint 测试。
- [x] V03 四个多任务配置和三个 split 的 multi-seed matrix dry-run。
- [x] V04 Ruff 与相关测试通过：22 passed。
