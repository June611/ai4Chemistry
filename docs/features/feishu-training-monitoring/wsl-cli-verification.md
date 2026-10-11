# WSL 飞书 CLI 真实读写验收

验收日期：2026-10-12。结论：WSL 已配置可用的飞书 CLI，使用现有用户登录可创建、
修改并读取多维表格和云文档。此次仅验证测试资源，没有启动正式训练。

## 本机配置

- 系统：WSL2，内核 `5.15.146.1-microsoft-standard-WSL2`。
- 命令：`/usr/bin/lark-cli`，官方 `@larksuite/cli`，版本 `1.0.97`。
- CLI 配置：`/home/june/.lark-cli/config.json`；应用凭证和用户 token 使用 CLI 的加密存储。
- 平台：`feishu`；实测身份：`user`；`auth status --json --verify` 返回 `verified=true`。
- 初次验证时 access token 已过期，服务器验证表明可以刷新；实际 API 调用成功。
- 权限包含 Base 创建、表/字段读取、记录创建/更新/读取，以及 Docx 创建/正文写入/读取。
- 本次没有打印、导出或复制 App Secret 和 access token。

此前只检查项目的 `FEISHU_*` / `LARK_*` 环境变量，未覆盖 CLI 的配置和加密登录缓存。
环境变量为空不能证明 WSL 的飞书 CLI 未配置；原先“还需新建应用才能开始”的判断已修正。

## 多维表格实测

[打开测试多维表格](https://my.feishu.cn/base/GmYJbiO30aQpbUsSk3Acgz7AnLg?table=tblTDdx9yU6yqJYX)

- 新建 Base：`WSL API 测试_训练监测_2026-10-12`。
- 新建数据表：`训练记录_API测试`，7 个字段。
- 新建一条记录：`WSL连接测试_非真实模型`。
- 修改同一条记录，然后以 record ID 读回，`has_more=false`、记录数为 1。

| 字段 | 创建值 | 修改后服务器读回值 |
|---|---:|---:|
| MSE | 0.25 | 0.125 |
| MAE | 0.5 | 0.25 |
| 训练耗时_s | 12.5 | 25 |
| 峰值内存_MiB | 128 | 256 |

全部指标是人为填写的权限测试数据，不是实际模型训练结果。该表保留在用户云空间，便于查看验收。

## 云文档实测

[打开测试云文档](https://my.feishu.cn/docx/S52zdfcglowRz9xJ1FjceVDcn4f)

- 创建空白测试文档：`WSL API 测试_文档创建修改_2026-10-12`，初始 revision=2。
- 读取初始正文，确认仅有标题。
- 追加测试说明及多维表格链接，revision=3。
- 将“文档测试状态：待修改。”替换为“文档测试状态：创建、修改、读取验证通过。”，revision=4。
- 读取最终正文，确认新文字存在、旧文字不存在，模拟数据说明和表格链接仍在。
- 创建、追加、修改接口成功；修改响应没有 warnings。

## 本地证据与边界

API 成功回执、文档各阶段读回 JSON、表格读回 NDJSON 和脱敏认证摘要保存在
`artifacts/feishu-monitor-smoke/wsl_cli_check_20261012/`，属于 Git 忽略的本机验收产物。

本次按官方 CLI 内置 `lark-shared`、`lark-base`、`lark-doc` 指引，使用 Base v3 shortcuts
和用户身份执行。后续可复用此登录，不需要重新安装或仅因环境变量为空而新建应用。

项目上一轮新增的 `python -m molgap.tracking` 当前仍采用独立 `FEISHU_*` 环境变量和
应用身份的 Bitable v1 HTTP 客户端，没有读取 CLI 登录缓存。因此本次成功证明的是
现有 CLI 的真实账号能力，不能替代该 Python 客户端与正式训练自动同步的端到端验收。
下一步如继续对接，应优先考虑复用已验证的 CLI 用户登录。
