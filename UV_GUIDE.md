# uv 环境速查（Python + PyTorch）

1. 在项目空目录中执行以下命令；`uv add` 会自动创建并同步 `.venv`：

```bash
uv init --package
uv python install 3.11
uv python pin 3.11
uv add "torch==2.6.0" --index pytorch-cu124=https://download.pytorch.org/whl/cu124
uv add chemprop pandas numpy scikit-learn
uv add --dev pytest ruff
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
2. 先用 `nvidia-smi` 查看驱动支持的最高 CUDA 版本；本机显示 12.4，所以选官方 `cu124` wheel。
3. 在 PyTorch 官方版本表确认该 Torch 版本提供目标 CUDA wheel；纯 CPU 环境改用 `https://download.pytorch.org/whl/cpu`。
4. 普通依赖用 `uv add X`，开发依赖用 `uv add --dev X`，其他用途用 `uv add --group NAME X`。
5. `uv add` 同时更新 `pyproject.toml`、`uv.lock` 和 `.venv`；克隆项目后只需运行 `uv sync --locked`。
6. 提交 `.python-version`、`pyproject.toml` 和 `uv.lock`，不要再混用手写 `requirements.txt` 或 `uv pip install`。
