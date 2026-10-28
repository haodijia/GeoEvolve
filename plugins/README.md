# Agent 组件

基础 Earth-Agent 不依赖具体组件。YAML 的 `plugins` 列表决定组件及执行顺序；空列表运行基础 agent。

| 名称 | 实现 | 作用 |
|---|---|---|
| `hkg` | `plugins.hkg.plugin:HKGPlugin` | 在问题前检索领域知识 |
| `far` | `plugins.far.plugin:FARPlugin` | 从失败轨迹生成下一轮反思 |

HKG 的知识库及 embedding 资源位于 `plugins/hkg/`；FAR 的记忆、反思提示词、轨迹与压缩策略位于 `plugins/far/`。运行状态写入 `<run>/result/plugins/`，组件源目录不保存实验状态。

## 新增组件

新增模块并继承 `plugins.base.AgentPlugin`：

```python
from plugins.base import AgentPlugin, PluginContext, PromptContribution

class MyComponent(AgentPlugin):
    async def setup(self, context: PluginContext) -> None:
        await super().setup(context)
        self.root = context.output_dir / "plugins" / "my_component"
        self.root.mkdir(parents=True, exist_ok=True)

    async def before_question(self, question: dict, query: str) -> PromptContribution:
        return PromptContribution(text="组件提供的额外上下文")
```

配置中直接指定模块路径，无需修改 agent 或内置注册表：

```yaml
plugins:
  - name: plugins.my_component.plugin:MyComponent
    options: {}
```

构造函数接收 `options`；有参数的组件应实现自己的构造函数并校验参数。生命周期：

1. `setup(context)`：每个数据集的每次重复运行各调用一次，获得配置、该 run 的 `result/` 目录和 actor 模型。
2. `before_question(question, query)`：返回附加提示词和日志元数据。不同组件不得写入同名元数据键。
3. `after_attempt(attempt, failed_results, chat_log_path)`：仅在存在失败题且还有下一轮时执行，返回 `prompt_tokens`、`completion_tokens`、`total_tokens` 三项用量。

调用 `after_attempt` 前，actor 的 `result/out/` 中间文件已经清空。组件应读取保存的轨迹和工具返回值；需要持久保存的组件状态写入自己的 `result/plugins/<组件名>/` 目录。反思用于重试时，必须考虑中间文件需要从原始输入重新生成。

组件调用模型时可使用 `agent.tokens.MeteredModel`。用量账本写入 `<run>/result/plugins/<组件名>/attempt_N/model_usage.jsonl`，保证被拒绝的输出与恢复运行成本也可追溯。组件内部用量单独记录，最终汇入 `experiment_summary.json` 的 `token_usage.plugins`，并与 actor 用量一起计入总成本。

恢复前 N 轮到新实验时，运行器复制前 N−1 轮的组件产物，并调用当前组件处理第 N 轮失败轨迹。FAR 按问题保存反思，原地继续不会重复生成已经保存的反思。

默认执行一次，不自动重跑题目。FAR 依赖多轮反思时，需在研究配置中显式设置 `max_attempts > 1`；其内部执行记录存入 `result/traces/`，最终由运行器生成通用 `results.jsonl`。评测插件只读取通用结果，不识别 FAR 的轮次。
