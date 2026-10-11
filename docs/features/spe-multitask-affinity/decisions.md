# SPE 多任务亲和力融合设计决定

## 物理定义

沿用项目 Phase 3 的列顺序 `[homo, lumo, delta_e]` 和关系
`delta_e = lumo - homo`。一致性残差定义为
`predicted_delta_e - (predicted_lumo - predicted_homo)`。

## 亲和力矩阵

每个任务从共享分子表示生成自己的隐藏表示。可学习 logits 形成 3×3 矩阵，按行 softmax 后，
第 i 行表示任务 i 从三个任务表示中吸收信息的比例。关闭亲和力时使用固定单位矩阵。

## 消融

保留既有单任务 SPE 基线；多任务部分采用 affinity × consistency 的 2×2 设计：

1. 多任务，无亲和力，无一致性；
2. 多任务，仅亲和力；
3. 多任务，仅一致性；
4. 多任务，同时使用亲和力和一致性。

所有组共享 tokenizer、encoder 容量、训练预算、split 和评价代码。
