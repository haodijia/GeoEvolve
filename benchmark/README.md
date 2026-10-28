# 数据集

代码与数据分别发布：GitHub 保存加载、下载、参考轨迹重建代码和版本锁；ModelScope 保存 `benchmark/earthbench/` 中的全部数据文件。

```text
benchmark/
  dataset.py                 通用数据集加载
  prepare.py                 下载、校验与发布清单生成
  build_ground_truth.py      EarthBench 参考轨迹重建
  earthbench.lock.json       数据仓库、版本与 manifest 校验值
  earthbench/                纯数据目录，不纳入 GitHub
    README.md
    LICENSE
    dataset.yaml
    manifest.json
    questions.json
    ground_truth.json
    data/question1/ ... question248/
    perception/
      observations.csv
      masks/
    provenance/
      SOURCES.md
      changes.json
      source_questions.json
      NOTICE
```

## 准备与校验

EarthBench-Fixed 使用 `haodijia/EarthBench-Fixed` 的 `2026-10-28` 发布。包含 248 道题、13,658 个输入文件、26 张掩膜、461 条固定感知观测与 1,378 次参考工具调用，输入约 16.6 GB。

```bash
# 按代码中的版本锁下载到 benchmark/earthbench/
uv run --locked python -m benchmark.prepare

# 离线校验全部发布文件，不生成数据目录内的临时报告
uv run --locked python -m benchmark.prepare --check
```

版本锁固定 manifest 的 SHA-256，manifest 固定每个文件的大小与内容。重复运行下载命令会校验已有文件并仅下载缺失或不匹配的文件；校验失败则报错。`--revision <commit>` 可指定上传后的 ModelScope commit，仍须通过同一个内容版本锁。`--report /path/to/report.json` 可将校验报告写到数据集之外。

运行路径统一相对于数据集根目录。`data/` 保存真实文件，`perception/masks/` 保存观测掩膜。输入只读；中间文件写入当前 run 的 `result/out/question<题号>/`，每题结束后删除本题目录。问题文件只包含 agent 输入，参考答案和轨迹单独保存在 `ground_truth.json`。

问题 148/149 的年度夜光栅格来自固定统计量合成，问题 214 的三个掩膜由 SAM2.1 Hiera Small 重建。来源、原始问题与修复记录保存在 `provenance/`，所有方法使用同一数据版本和固定观测。

## 发布到 ModelScope

仓库继续使用 `haodijia/EarthBench-Fixed`。将 **`benchmark/earthbench/` 的内容** 放在数据仓库根目录，根目录直接出现 `README.md`、`dataset.yaml`、`data/` 和 `perception/`，不再套一层 `earthbench/` 或 `benchmark/`。

更新已有远程仓库时，先移除旧布局文件，再上传完整新目录，避免旧 `benchmark/`、代码、缓存快照与新文件混在一起。保留数据许可、LFS 配置和 `provenance/`。上传后使用代码中的下载命令，在干净位置核对发布内容。

发布完成后，将代码侧 `earthbench.lock.json` 的 `revision` 固定为本次 ModelScope commit。数据包内容不变，下载同时校验 commit 对应的 manifest 与文件内容。

数据维护后重新生成清单和代码侧版本锁：

```bash
uv run --locked python -m benchmark.prepare --seal \
  --repo haodijia/EarthBench-Fixed --version 2026-10-28-tools104 --revision master
```

参考轨迹重建使用归档的源问题文件，普通运行和评测直接使用发布的参考文件：

```bash
uv run --locked python -m benchmark.build_ground_truth
```

## 接入新数据集

新增 `benchmark/<name>/dataset.yaml`，名称使用小写字母开头及小写字母、数字、下划线：

```yaml
name: example
loader: benchmark.dataset:load_questions
questions: questions.json
ground_truth: ground_truth.json
data: data
source_paths:
  data: data
```

标准问题文件是 JSON 列表，每项包含：

```json
{
  "question_id": "1",
  "auto": "自主规划问题",
  "instruct": "指导式问题",
  "data": "data/question1/input.tif",
  "choices": ["选项一", "选项二"]
}
```

开放题使用 `choices: null`。问题 ID 唯一。加载器只返回问题；其他原始格式可声明自己的 `module:function` 加载器。

参考文件是 JSON 列表，每项包含 `question_index`、`final_answer` 和 `tool_calls`；工具调用包含 `name`、`input`。数字问题 ID 在参考文件中使用 `question1` 等形式。固定版本数据可声明 `manifest`，固定感知环境可声明 `perception_cache`。将数据集名称加入配置的 `datasets`，或通过 `--datasets earthbench example` 选择。
