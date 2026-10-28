# HKG embedding 权重

HKG 使用 `sentence-transformers/all-mpnet-base-v2` 的固定版本。基础 agent 和 FAR 不需要本地模型权重。

```bash
uv sync --locked --extra hkg
uv run --locked --extra hkg hf download sentence-transformers/all-mpnet-base-v2 \
  --revision e8c3b32edf5434bc2275fc9bab85f82640a19130 \
  --local-dir plugins/hkg/models/all-mpnet-base-v2
```

配置的 `plugins[].options.model` 指定权重目录。所有 HKG 对比使用同一版本。权重不纳入 Git。
