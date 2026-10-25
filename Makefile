# 常用任务入口（uv + ruff + pytest）
.PHONY: help setup lint format format-check test test-fast docs-check clean

help:
	@printf '%s\n' \
	  'make setup        # 安装锁定依赖（含 qm9 extra）' \
	  'make lint         # ruff check' \
	  'make format       # ruff format + fix imports' \
	  'make format-check # 只检查格式' \
	  'make test         # 全量测试' \
	  'make test-fast    # 跳过慢速管线测试' \
	  'make docs-check   # 校验 feature 文档必备文件' \
	  'make clean        # 清理缓存（不动 artifacts/）'

setup:
	uv sync --locked --extra qm9

lint:
	uv run --no-sync ruff check .

format:
	uv run --no-sync ruff check --fix .
	uv run --no-sync ruff format .

format-check:
	uv run --no-sync ruff format --check .

test:
	uv run --no-sync pytest -q

test-fast:
	uv run --no-sync pytest -q -k 'not pipeline and not smoke'

docs-check:
	@fail=0; for d in docs/features/*/; do \
	  for f in README.md decisions.md; do \
	    [ -f "$$d$$f" ] || { echo "缺少 $$d$$f"; fail=1; }; \
	  done; \
	done; exit $$fail

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
