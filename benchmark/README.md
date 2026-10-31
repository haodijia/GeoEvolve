# 数据集

GitHub 保存工具、加载器、维护命令与版本锁；ModelScope 的 `haodijia/EarthBench-Fixed` 保存数据。
本版 `2026-10-31-contracts-v3` 保留 248 题题干及 104 个工具接口，统一工具执行、569 条固定观测、参考链和答案。

```bash
uv run --locked python -m benchmark.prepare
uv run --locked python -m benchmark.prepare --check
uv run --locked python scripts/validate_mcp_tools.py --replay-ground-truth
```

`benchmark/earthbench.lock.json` 固定 ModelScope commit 和版本。`manifest.json` schema 2
记录每个文件的相对路径与大小，不新增内容哈希。下载检查缺失文件与大小；离线检查还检查多余文件和软链。
切换发布清单时重新下载输入，避免同大小的元数据修订被跳过；旧清单中的废弃文件会被移除。

数据根目录包含 `questions.json`、`ground_truth.json`、`dataset.yaml`、`data/`、
`perception/observations.csv`、`perception/masks/` 和 `provenance/`。输入只读，临时输出写入每题独立工作目录。
感知缓存已整合进数据集，无需外置补丁或模型服务。

`questions.json` 只含 agent 输入，参考与修订证据不进入提示词。原版文件归档在
`provenance/previous_release/`，补充缓存、元数据、答案和参考链修订均保留来源。
问题 148/149 沿用此前的合成栅格声明，不宣称原始遥感影像。

## 维护发布

参考依赖计划保存在数据集 `provenance/reference-plan.json`，每一步通过真实工具执行。
`provenance/answer-review.json` 保存从返回值推导答案的证据；回放同时验证这些证据。
旧的仅转换源 dialogs、复用旧输出的参考生成器已移除。

```bash
uv run --locked python -m benchmark.rebuild_references --work-dir /path/to/new-replay
uv run --locked python -m benchmark.prepare --seal \
  --repo haodijia/EarthBench-Fixed --version 2026-10-31-contracts-v3 --revision master
```

审核回放输出后更新发布参考，完成 MCP 全量验证，再上传数据根目录内容。上传后将代码版本锁的
revision 固定到实际 ModelScope commit，不使用浮动 master 进行正式评测。

## 新数据集

在 `benchmark/<name>/dataset.yaml` 声明 `name`、`loader`、`questions`、`ground_truth`、`data`
与 `source_paths`。标准问题包含 `question_id`、`auto`、`instruct`、`data`、`choices`，开放题 choices 为 null。
参考包含 `question_index`、`final_answer`、`tool_calls`，每个调用包含 name、input、output。
