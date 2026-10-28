# Earth-Bench pro max

用于复现 Earth-Agent、评测遥感任务、组合和比较自研组件的实验框架。基础 agent、数据集、组件和评测独立组织，所有实验使用 YAML 配置。

```text
benchmark/                 数据集注册与加载
  earthbench/              EarthBench-Fixed 的问题、数据、参考轨迹和观测缓存
configs/                   每个实验的配置
agent/                     基础 Earth-Agent、运行器、MCP 配置与工具
plugins/                   可选组件及其资源
  hkg/                     RSMeM 知识检索
  far/                     RSMeM 失败反思与轨迹策略
scripts/                   批量启动与工具校验
evaluation/                统一评测入口、提取器与指标模块
runs/                      本地运行产物，不纳入 Git
```

## 安装与数据准备

使用 Python 3.10–3.12 和 uv，依赖由 `pyproject.toml` 与 `uv.lock` 固定。

```bash
uv sync --locked
uv run --locked python -m benchmark.prepare
uv run --locked python -m benchmark.prepare --check
```

EarthBench-Fixed 约 16.6 GB，包含 248 个问题。`benchmark/earthbench.lock.json` 固定数据版本与整包清单校验值，下载后逐文件校验。`benchmark/earthbench/` 是独立发布到 ModelScope 的纯数据目录，数据与模型权重不纳入 Git。数据布局和接入方式见 [benchmark/README.md](benchmark/README.md)。

## 运行

通过环境变量提供模型 API key。配置中的 `api_key_env` 指定变量名，密钥不写入配置快照。

```bash
export OPENAI_API_KEY='your-api-key'

# 预检配置与题目选择，不调用模型
bash scripts/run.sh --config configs/deepseek_v4flash_baseline.yaml --sample 10 --dry-run

# 基础 Earth-Agent
bash scripts/run.sh --config configs/deepseek_v4flash_baseline.yaml

# 同一个 run 内最多同时运行 4 道题
bash scripts/run.sh --config configs/deepseek_v4flash_baseline.yaml --concurrency 4

# 同一配置只加载 FAR，运行前 10 题
bash scripts/run.sh --config configs/deepseek_v4flash_baseline.yaml --plugins far --sample 10

# 关闭全部插件
bash scripts/run.sh --config configs/deepseek_v4flash_baseline.yaml --plugins

# 组合组件；默认每题只执行一次
bash scripts/run.sh --config configs/deepseek_v4flash_baseline.yaml --plugins hkg far
```

启动脚本按所选插件安装 HKG 的额外 Python 依赖。启用 HKG 前需准备[固定版本的 embedding 权重](plugins/hkg/models/README.md)。FAR 和基础 agent 不需要本地模型权重。

`--concurrency N` 设置每个 run 的题目并发上限，省略时读取配置的 `concurrency`，默认 `1`。每题使用独立 agent 和 MCP 工具配置；结果完成后立即保存，按原题目顺序写入结果文件。

`--datasets earthbench <另一个已注册的数据集名>` 顺序运行多个数据集；省略时使用配置中的列表。当前附带 EarthBench-Fixed。`--sample N` 选择每个数据集的前 N 题，`sample: null` 运行全部题目；超出数据集大小会报错。`--plugins` 覆盖插件列表，保留配置中同名插件的参数；新指定的内置插件使用自身默认参数。插件按列表顺序执行。

每次启动为每个数据集生成一个独立 run。目录名为 `数据集_模型简称_日期时间`，数据集使用注册名 `earthbench`，模型简称由配置中的 `model_label` 指定。

```text
runs/earthbench_deepseek_20261028_103829/
  result/
    run.json
    experiment_config.yaml
    dataset_snapshot.json
    run.chat
    results.jsonl
    experiment_summary.json
    out/
      question1/
      question2/
    traces/
    plugins/
  evaluation/
    extracted/
    metrics/
    summary.json
    protocol.json
    state.json
```

每题使用独立的 `result/out/question<题号>/`，执行后保存记录并删除本题目录；失败、超时和中断也执行清理，不影响其他正在执行的题。默认 `max_attempts: 1`，执行失败只记录，不自动重新跑题；模型 SDK 的隐式请求重试已关闭。`result/results.jsonl` 每题保存一条最终结果，`experiment_summary.json` 汇总实际成本。内部执行细节归入 `traces/`，插件状态归入 `plugins/`。

手动继续中断的 run，跳过已完成题目：

```bash
bash scripts/run.sh --config <原配置> --continue-attempt <run目录>
```

FAR 多轮反思属于显式研究设置，需要在独立配置中设置 `max_attempts > 1`。其内部迭代只影响方法执行，最终仍提交同一格式的结果包，评测无需选择轮次。

## LangSmith 远程查看

默认 `langsmith: false`，通过 `--langsmith` 开启，或通过 `--no-langsmith` 关闭。命令行优先于配置；关闭时，即使环境变量设置了 tracing，也不会为当前 run 上传轨迹。

```bash
export LANGSMITH_API_KEY='your-langsmith-api-key'
bash scripts/run.sh --config configs/deepseek_v4flash_baseline.yaml \
  --sample 10 --concurrency 4 --langsmith
```

模型 API key 仍使用 `OPENAI_API_KEY`。启用 tracing 必须提供 `LANGSMITH_API_KEY`；密钥不写入配置快照。`LANGSMITH_ENDPOINT` 可选择自己的服务地址或账户所在区域，`LANGSMITH_WORKSPACE_ID` 可选择工作空间。

每个本地 run 自动对应一个独立的 LangSmith 项目，项目名与 `runs/` 下的目录名完全相同，例如 `earthbench_deepseek_20261028_144757`。Qwen、DeepSeek 和同一模型的不同实验各自独立，选择对应项目即可查看该次实验，无需手动筛选混合列表。在项目内按 `question_id`、`attempt` 和 `model_role` 区分题目、执行次数及 actor/reflector/compressor。继续中断的 run 时沿用同名项目。并发题目各自有独立调用链；退出前会等待 tracing 队列发送完成。

开启后，问题提示、模型响应、工具入参与返回值会发送到 LangSmith，用于远程查看调用链、耗时与错误。本地运行结果和 evaluation 仍保存在 `runs/`。接入方式见 [LangSmith 官方文档](https://docs.langchain.com/langsmith/trace-with-langgraph)。

## 配置与组件

`configs/` 仅保留 DeepSeek 和 Qwen3 8B 两份 baseline 配置。通过 `--plugins` 选择组件组合，无需为每种组合维护单独配置。组件参数归各自插件管理。

```yaml
model_label: deepseek
model_name: deepseek-v4-flash
base_url: https://api.deepseek.com
api_key_env: OPENAI_API_KEY
datasets: [earthbench]
sample: null
concurrency: 1
langsmith: false
max_attempts: 1
plugins:
  - name: hkg
    options:
      knowledge: plugins/hkg/knowledge.json
      model: plugins/hkg/models/all-mpnet-base-v2
  - name: far
    options:
      reflector: actor
      trajectory_strategy: full_tool_result_only
      reflection_strategy: default
agent_config_path: agent/mcp.json
output_base_dir: runs
```

FAR 的 `reflector: actor` 明确复用执行模型，也可提供独立的模型配置。使用 `summary_ftro` 时，需在 FAR 的 `options` 中配置 `summary_ftro_source` 和 compressor。新增组件只需实现生命周期接口，并在 YAML 中写入 `module:Class`。见 [plugins/README.md](plugins/README.md)。

## 评测

```bash
uv run --locked python -m evaluation --list
uv run --locked python -m evaluation --run runs/earthbench_deepseek_20261028_103829

uv run --locked python -m scripts.validate_mcp_tools --replay-ground-truth
```

评测自动读取所选 run 的数据集和最终结果，提取结果与指标写入对应的 `evaluation/` 子目录。指标以本次运行的题目数为分母；路径归一化、token 成本及数据来源见 [evaluation/README.md](evaluation/README.md) 和 [评测协议](evaluation/PROTOCOL.md)。MCP 工具说明见 [agent/tools/README.md](agent/tools/README.md)。

## 来源与许可

项目基于 [RSMeM](https://github.com/AI9Stars/RSMeM) 与 [Earth-Agent](https://github.com/opendatalab/Earth-Agent) 的代码进行修复和重构。原始许可与署名保留在 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
