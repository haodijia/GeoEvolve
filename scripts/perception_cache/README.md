# 离线补缓存环境

此目录仅用于离线生成待审核的感知观测；正常评测不加载这些模型，也不启动服务。
数据筛选、生成、审核和冻结流程见 [缓存修复说明](../../docs/perception-cache-repair.md)。

本机复用已有 uv 环境 `/home/dijkstra/model/.venv-earthbench`，模型放在
`/home/dijkstra/model`，生成和审核材料放在 `/home/dijkstra/data/GeoEvolve/cache-repair/`。
ChangeOS 的本地源码位置在 `pyproject.toml` 中显式声明。

```bash
cd scripts/perception_cache
source ./env.local.sh
uv sync --locked --inexact --no-build-isolation-package sam2
uv run --locked --no-sync python ../repair_perception_cache.py detections \
  --directory /home/dijkstra/data/GeoEvolve/cache-repair/v1
```

`classifications`、`masks` 和 `aliases` 分别生成分类、掩膜与语义别名候选。
候选不能直接用于评测，必须先审核再通过 `freeze` 安装冻结补丁。
