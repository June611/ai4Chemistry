# 中断恢复与自动审批

2026-10-12 本地检查：`/home/june/.local/bin/codex` 为 0.162.1，用户配置顶层已包含：

```toml
approval_policy = "on-request"
approvals_reviewer = "auto_review"
sandbox_mode = "workspace-write"
```

`codex doctor --json` 确认配置解析及加载成功；项目为 trusted，自动审批相关 guardian_approval 功能已启用。该诊断不检查当前会话的临时覆盖。

进一步只读取本项目最近 CLI 会话的 turn_context 权限字段：2026-10-11T17:13:10.434Z 的记录仍为 `approvals_reviewer=user`、`approval_policy=on-request`、workspace-write。这解释了为什么配置文件已改好，当前旧会话仍可能弹出人工审批。需要切换当前会话模式或显式恢复，不能仅检查配置文件。

当前 CLI 通过 `/permissions` 选择 **Approve for me**，或重新进入本项目并显式恢复：

```bash
cd /home/june/deepLear_zone/Chem_proj/chempro
codex resume --last --approve-for-me
```

该参数已在本地 `codex resume --help` 中核验。自动审批对需要跨越沙箱边界的请求作批准或拒绝判断，不改变沙箱边界，也不是无条件批准。[官方说明](https://learn.chatgpt.com/docs/sandboxing/auto-review)

用户提到 Shift 误触导致中断，尚未确认具体组合键，不能判断根因。CLI 可通过 `/keymap` 查看及重绑动作；运行中 Tab 可排队后续消息，Enter 会向当前执行插入指令。[官方快捷键与 keymap](https://learn.chatgpt.com/docs/developer-commands?surface=cli)

本次服务器训练由 nohup 后台进程运行，日志与状态写入 artifacts，终端会话中断不会直接终止训练。恢复前检查 status.json 及进程状态；遇到仍在运行的训练不得重复启动。运行器发现未收尾的 running 状态会要求先检查，不覆盖产物。

服务器入口和监测器独立上传，不依赖对话持续在线。补测/合并入口：

```bash
python scripts/complete_smoke_resource_report.py \
  --output artifacts/one_epoch_smoke_20261012 --runner-pid 11942
```

此命令本轮已后台启动；PID 仅是本次运行值，后续启动时必须替换为实际 runner PID。
