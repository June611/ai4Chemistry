# WSL 通过 SSH 反向代理加速蓝耘访问 GitHub

## 本次验证结果

- WSL 中 `127.0.0.1:7890` 不可达；实际可用的 Clash 入口由环境变量给出，为 `172.23.96.1:7897`。
- 蓝耘远端端口 `127.0.0.1:17890` 已成功转发到上述 Clash 入口。
- 仓库已克隆到 `/root/lanyun-tmp/evan/ai4Chemistry`，HEAD 为 `8fda416702b46e0ca6ee8b88de1a9860c249d00e`。

## 手动操作

先在 WSL 查看实际代理地址，不要仅凭 Clash 界面中的端口猜测：

```bash
env | grep -iE '^(http|https|all)_proxy='
curl -I --max-time 15 -x http://172.23.96.1:7897 https://github.com
```

在 WSL 终端建立反向转发。该终端必须保持运行：

```bash
ssh \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -R 127.0.0.1:17890:172.23.96.1:7897 \
  -p 36398 root@jllink.lanyun.net
```

登录蓝耘后先测试代理，再克隆：

```bash
curl -I --max-time 20 -x http://127.0.0.1:17890 https://github.com
mkdir -p /root/lanyun-tmp/evan
git -c http.proxy=http://127.0.0.1:17890 clone \
  https://github.com/June611/ai4Chemistry.git \
  /root/lanyun-tmp/evan/ai4Chemistry
```

以后更新仓库时，在反向转发仍然有效的 SSH 会话中运行：

```bash
git -C /root/lanyun-tmp/evan/ai4Chemistry \
  -c http.proxy=http://127.0.0.1:17890 pull --ff-only
```

如果希望隧道与远端操作分开，在第一个 WSL 终端用 `ssh -N` 建立隧道，再用第二个终端正常登录服务器。退出第一个 SSH 进程后，`17890` 代理立即失效。

若以后 Clash 的监听地址或端口变化，将命令末尾的 `172.23.96.1:7897` 替换为 WSL 中实际可连接的地址。只有当 `curl -x http://127.0.0.1:7890 ...` 在 WSL 内测试成功时，才能使用 `127.0.0.1:7890`。远端监听保持为 `127.0.0.1:17890`，避免将个人代理暴露给公网。
