# 数据独立迁移

代码、配置、来源锁定和审查文档用 Git 管理。约 500 MiB 的 `data/` 单独打包后用 SCP、
网盘或对象存储传输；不要把压缩包、几何数据、权重或上游仓库添加到 Git。
本次只准备本地数据包，没有上传到任何网盘或外部存储。

2026-10-11 本地包：`artifacts/qm9_data_20261011.tar.gz`。
可追踪清单：`docs/data/qm9_data_20261011.manifest.json`，记录每个文件的字节数与 SHA-256。
包包括整个本地 `data/`，含历史 CN 子集和测试子集；正式协议只读 `qm9_full` 三套 split
及三份 geometry，不把历史子集混入训练。

构建新日期的包（拒绝覆盖已有包/清单）：

```bash
uv run python scripts/package_qm9_data.py \
  --archive artifacts/qm9_data_20261011.tar.gz \
  --manifest docs/data/qm9_data_20261011.manifest.json
```

上传这个包和相邻 `.sha256` 文件即可。代码仓库拉取后，在服务器项目根目录放入包并解压：

```bash
mkdir -p artifacts
# 将压缩包和 .sha256 放在 artifacts/，例如用 scp 或网盘下载。
(cd artifacts && sha256sum -c qm9_data_20261011.tar.gz.sha256)
# 在新项目目录或已备份旧 data 的目录中解压，tar 会覆盖同名文件。
tar -xzf artifacts/qm9_data_20261011.tar.gz
uv run python scripts/package_qm9_data.py \
  --archive artifacts/qm9_data_20261011.tar.gz \
  --manifest docs/data/qm9_data_20261011.manifest.json --verify
uv run python scripts/audit_qm9_split_seeds.py
uv run python scripts/audit_paired_seed_protocol.py
```

校验清单包含全部源文件；缺文件或哈希不符即失败。解压前检查压缩包 SHA-256，解压后校验
逐文件 SHA-256，避免网盘传输、手工修改和不完整复制破坏既有划分。

.gitignore之外的旧data索引清理已完成：34个数据条目取消跟踪，所有本地文件保留。`.gitignore` 只对未跟踪文件生效，
不能自动消除历史记录。先确认独立包验证通过，再对已跟踪数据执行 `git rm --cached` 并保留
`.gitkeep`，这只取消版本跟踪，不删除本地数据。本轮已完成索引清理并随代码交接提交；上传方式仍待选择。

历史提交中已有的 CSV 仍会影响完整 clone 大小。新的提交取消跟踪后，新服务器可先使用
`git clone --depth 1` 获取当前快照，再搬运数据。清理历史需要单独安排，不使用 force push。

GitHub 单文件超过 50 MiB 会告警、超过 100 MiB 会被阻止；本地 `gdb9.sdf` 约 225 MiB。
参考 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
独立数据包便于跨服务器复用；Git LFS 是备选，但要考虑配额和团队成员的 LFS 环境。
