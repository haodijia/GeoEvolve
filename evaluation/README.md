# 评测

所有方法提交同一种最终结果包，主入口执行“提取 → 指标计算”。评测读取最终答案、提交轨迹、累计调用与成本，不读取方法内部的执行轮次。

```bash
uv run --locked python -m evaluation --list
uv run --locked python -m evaluation --run runs/earthbench_deepseek_20261028_103829

# 选择指标
uv run --locked python -m evaluation \
  --run runs/earthbench_deepseek_20261028_103829 \
  --metrics answers trajectory tokens

# 输入、代码和输出校验值均一致时复用报告
uv run --locked python -m evaluation --run runs/earthbench_deepseek_20261028_103829 --resume
```

无需指定数据集、参考文件、结果目录或轮次。baseline 默认每题执行一次；FAR 如在研究配置中显式启用内部迭代，也提交相同的最终结果包。只有执行完成的 run 可以评测，执行失败的题目作为结果记录保留并计分。

## 目录结构

```text
runs/earthbench_deepseek_20261028_103829/
  result/
    run.json
    experiment_config.yaml
    dataset_snapshot.json
    results.jsonl                      通用结果，每题一条
    experiment_summary.json            通用运行汇总与实际成本
    run.chat
    out/                               每题结束清空
    traces/                            内部执行记录，供审计和手动恢复
    plugins/                           插件状态与用量账本
  evaluation/
    extracted/
      answers.json
      trajectories.json
    metrics/
      answers.json
      trajectory.json
      tokens.json
    summary.json
    protocol.json
    state.json
```

`traces/` 和 `plugins/` 保存方法内部细节；评测不依赖这些目录的组织方式。输入数据、题目集合和成本校验不一致时直接报错。评测不打开 actor 的临时输出文件。

## 统一结果契约

`result/results.jsonl` 每题一条记录：

| 字段 | 含义 |
|---|---|
| `question_id` | 题目 ID，在 run 内唯一 |
| `answer` | 最终提交的答案 |
| `execution_error` | 最终提交是否执行失败 |
| `tool_calls` | 产生最终答案的工具轨迹 |
| `cumulative_tool_calls` | 为该题实际执行的全部工具调用，包括失败调用和内部迭代 |
| `output_roots` | 轨迹中出现的输出根目录，用于参数归一化 |
| `token_usage` | 该题累计 actor 用量，包含产生最终结果之前的内部执行 |

`experiment_summary.json` 中的 `token_usage` 分别记录 `actor`、`plugins` 和 `total`。插件的反思、压缩和格式校正成本均计入总成本。最终准确率与最终轨迹衡量提交结果；累计调用和全部角色的用量衡量得到该结果的实际成本。

每种方法负责生成结果包。baseline 的内部记录只有一次执行；FAR 多次执行时保留最后一次答案和提交轨迹，累计调用及 actor 成本覆盖全部内部执行。FAR 的轮次编号只存在于内部记录中。

## 代码结构与扩展

```text
evaluation/
  __main__.py              统一入口与流程编排
  context.py               校验通用结果包
  protocol.py              数据来源和路径归一化
  extraction/
    answers.py             提取最终答案
    trajectories.py        提取提交与累计轨迹
  metrics/
    answers.py             答案、覆盖率、失败率和工具调用成本
    trajectory.py          工具顺序、匹配和参数准确率
    tokens.py              实际 actor/plugin token 成本
```

提取器和指标由主入口自动发现。辅助模块使用下划线开头的文件名，不参与发现。

新增提取器，在 `extraction/` 增加模块：

```python
def extract(records: list[dict]) -> list[dict]:
    return [{"question_id": record["question_id"]} for record in records]
```

`records` 就是通用最终结果列表。新增指标，在 `metrics/` 增加模块：

```python
from evaluation.context import MetricContext

REQUIRES = ("answers",)


def compute(context: MetricContext) -> dict:
    return {"summary": {"prediction_count": len(context.extracted["answers"])}}
```

`context.run` 提供配置、数据集、最终结果、参考答案与运行汇总。指标返回对象必须包含 `summary` 字典，其他字段可保存逐题详情。完整对象写入对应指标文件，`summary` 汇入 run 的评测汇总。新增模块后可用 `--metrics` 选择，也会加入默认的全指标评测。

默认重新计算。`--resume` 仅在输入、代码、环境与全部输出校验值一致时复用报告；改变指标选择后会清理旧选择的衍生产物。

## 指标口径

准确率、覆盖率、失败率和轨迹得分均以本次 run 的题目集合为分母。题目 ID 与保存的数据集快照核对，参考答案和轨迹只取这一集合。跑两题，分母就是 2。

| 指标 | 含义 |
|---|---|
| Accuracy | 正确答案 / 本次运行的问题数 |
| Coverage | 已提交预测 / 本次运行的问题数 |
| Fail rate | 执行失败或空答案 / 本次运行的问题数 |
| Tool_Any_Order | 预测包含的不同参考工具名占比 |
| Tool_In_Order | 按顺序匹配的参考工具子序列前缀占比 |
| Tool_Exact_Match | 工具名逐步匹配的前缀长度 / 参考长度 |
| Parameter | 工具名及完整参数逐步匹配的前缀长度 / 参考长度 |
| Tool-call ratio | 提交轨迹或累计工具调用数 / 参考调用数 |

答案使用最后一对 `<Answer>value<Answer>` 标签，也接受 `</Answer>` 闭合。格式无效和执行失败的答案计零。路径归一化保留文件名、数值和列表顺序。批量操作按一次工具调用计算。

`tokens.json` 自动计算累计 actor/plugin 成本、插件占比和 `accuracy / log10(tokens)`。零 token 时占比为 `null`，总 token 不大于 1 时对数效率为 `null`。模型返回的 token 数不是货币账单。

EarthBench-Fixed 的数据来源、修复规则和公式见 [PROTOCOL.md](PROTOCOL.md)。参考轨迹重建是数据集维护命令：

```bash
uv run --locked python -m benchmark.build_ground_truth
```
