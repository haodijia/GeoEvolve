# GeoEvolve 复现检查与修复

日期：2026-10-24。上游为 `haodijia/GeoEvolve`，本机起始提交 `638e33a`。本轮修改记录于本地 Git，未推送；没有发起在线 LLM 实验。用户提到的 `4daf40b` 不在这个仓库的现有提交历史中。

## 当前采用的基准

以官方 EarthBench 的题目、标准答案及预计算感知观测为基准，不采用或部署 RSMeM 的 GPU HTTP 服务。RSMeM 的方法组件在这套共同工具环境上比较。保留执行修复和官方轨迹需要的六个接口，不因原版候选工具缺少观测就另行引入模型服务。缺失条目严格报错并记录环境失败，不运行替代模型。

## 已确认与已修复

- 原版 [RSMeM](https://github.com/AI9Stars/RSMeM/blob/main/tools/Perception.py) 通过 HTTP 调用感知推理服务；GeoEvolve 的对应实现使用固定观测缓存。这两种工具环境并不完全等价，缓存只覆盖已记录的图像、提示词和框。
- 工具输出改为每次实验独立的绝对 `<experiment_dir>/out`，不再共享 `tmp/out`。配置文件移除固定输出目录；runner 统一注入。时间戳包含微秒，新实验拒绝复用已有目录。
- 修复 Matplotlib 新版本删除 `cm.get_cmap` 引发的 `grayscale_to_colormap` 故障。
- runner 保存结构化工具轨迹，保留失败和未完成调用；评测不再用日志正则推测调用。每题总时限通过 `asyncio.wait_for(..., timeout=300)` 生效。
- 官方四个工具指标公式保留。所有 248 题参与平均，缺答记零；只请求第 N 轮也会读取此前所有轮次。参数比较统一存储根目录，同时保留文件名和实际参数差异。
- 末轮工具调用比和累计调用比分别报告。修复 actor 使用标准 LangChain usage metadata 的统计，重试/压缩/反思的实际返回 token 在解析输出前持久化；断点续跑继承此前成本。移除了“末轮 actor token 减首轮 actor token 等于反思成本”的错误计算和固定反思兜底文本。

## 工具数量口径与官方接口不一致

官方 Earth-Agent 与原版 RSMeM 当前源码各注册 104 个工具。GeoEvolve 拉取时就已注册 110 个，新增暴露 `calculate_ndvi`、`calculate_ndwi`、`calculate_ndti`、`calculate_nbr`、`argmax`、`index_to_date_range`；没有删除官方工具。本轮修复没有新增这六个接口。

进一步核对原始 `question.json` 的顶层 `dialogs`，六个名字全都已出现在官方参考轨迹中：NDVI 3 次、NDWI 13 次、NDTI 11 次、NBR 16 次、argmax 1 次、index_to_date_range 1 次，共 45 次。前四个在官方源码中有内部函数，但没有注册给 agent；后两个在公开工具模块中没有对应接口。因此公开参考轨迹与公开 MCP 接口本身不一致，扩展至 110 个接口与补齐这些调用相对应；仅有当前初始提交，无法还原此前逐次修改的动机。

本项目以修复 EarthBench 为目标，保留六个接口，形成 110 接口的修复版环境。四个指数计算函数去除名称差异后，其 AST 与原版 RSMeM 内部函数完全一致；另外两个接口通过实际 MCP 调用，返回值与官方 q183、q110 示例一致。“110/110 工具检查通过”指该修复版环境。后续所有方法共用同一环境与评测器，并披露对公开版本的修复。证据见本机 `tmp/review/official_unregistered_calls.json` 和 `six_tool_semantics.json`。

## 感知观测来源

CSV 从 219 条扩展为 461 条：MSCN 172、RemoteCLIP 172、SM3Det 57、InstructSAM 24、RemoteSAM 10、SAM2 9、ChangeOS 17。

合并优先级：有效官方观测 > 已有 GeoEvolve 补缺观测 > 本机 SAM2 补缺观测。六处 InstructSAM 数值冲突已恢复官方原值，均经实际 MCP 调用验证。21 条缺少有效结果或可读取掩膜的官方记录未直接采用，其缺失产物由已注明来源的修复资产补齐。详细来源在 `/home/dijkstra/data/earth_bench_perception/merge_provenance.json`。

172 条 RemoteCLIP 直接恢复自官方 CSV。本机只有 ViT-B/32，原 RSMeM 指定 ViT-L/14，因此未换用本机模型重算这些结果。q214 A/B/C 的三张缺失 SAM2 掩膜使用本机 SAM2.1 Hiera Small 和官方框重建，来源在 `sam2_provenance.json`。重建产物位于 `/home/dijkstra/data/earth_bench_perception/sam2`，CSV 记录绝对路径，迁移机器时需要同步并改路径。

**Strip-R-CNN 没有官方缓存观测，不能执行；本项目不为它部署在线服务。** 官方 Earth-Agent 公开实现同样只读 CSV，当前官方 454 条 CSV 中没有任何 Strip-R-CNN 记录，查不到时返回 `Failed to call model`；这不仅是本机缺权重，而是公开版本也存在接口与可用观测不一致。RSMeM 则改为 HTTP POST 到 `STRIP_RCNN_DETECT_URL`，需要额外部署服务。不能据此推断作者论文运行环境也缺少该工具。未用其他检测器冒充。未知图像、提示词、框也会严格报缓存缺失。后续统一比较必须让所有方法使用同一套观测与掩膜。

## 评测口径

| 指标 | 当前含义 |
|---|---|
| Accuracy | 正确答案 / 248，漏答和运行失败记零 |
| Tool_Any_Order | 命中的 GT 不同工具名 / GT 不同工具名数 |
| Tool_In_Order | 按顺序找到的 GT 工具序列前缀 / GT 长度，可跳过额外预测调用 |
| Tool_Exact_Match | 同位置工具名前缀 / GT 长度，对应官方 `trajectory_step_wise_score` |
| Parameter | 工具名和完整参数均匹配的前缀 / GT 长度，归一化存储根目录 |
| Call ratio | 分别报告末次轨迹、累计全部重试的调用数 / GT 调用数 |
| Tokens | 逐轮累计 actor + 生成下一轮所需的反思与压缩调用，格式重试同样计入 |

另一个 `contains_all_tool_calls_in_order_exact_match` 是除以较长序列的附加诊断，不能代替官方 Tool_Exact_Match。详细公式、命令和轮次成本边界见 [evaluation/README.md](evaluation/README.md)。token 统计基于服务商实际返回的 usage；请求没有返回可核对的 usage 时不视作免费调用。

原题目和标准答案未修改。当前可执行 GT 是此前已修复的 248 题 / 1,378 调用，**与未经修复的官方轨迹不是同一协议**；路径归一化、全题分母与严格答案格式也需要披露。公开归档中的 polished answers 只能用于验证评测流程，不能视为本次模型运行。

## 数据边界

官方下载数据保留在 `/home/dijkstra/data/earth_bench`，未写入修补文件。`benchmark/data` 指向 `/home/dijkstra/data/earth_bench_geoevolve_review`，其中官方文件通过链接使用，q148/q149 额外有四张合成栅格。

`benchmark/reconstruct_missing_data.py` 根据硬编码均值、有效像素数和热点数合成夜光栅格，并未恢复真实像素或空间分布。正式结果应披露，并对这两题单独报告。缓存掩膜重建同样需要披露，不能把当前环境描述为所有官方感知输出的原样复现。

## 验证证据

以下 JSON 和日志位于本机 `tmp/review/`，均为检查产物：

| 检查 | 结果 / 证据 |
|---|---|
| 原有 GT 重建与已提交 GT 比较 | 248 题、1,378 调用一致；最终答案 248/248 与官方 whitelist 一致 |
| 修复版全量 GT 回放 | 248/248、1,378 调用执行通过，随后 110/110 工具验证通过；`final_benchmark.log`、`final_benchmark_validation.json`。该次回放早于六处计数恢复官方值；恢复后六处经 MCP 单独验证，未重复全量回放 |
| 修复后现有工具验证 | 110/110 工具通过，`updated_tools.log` |
| 实际缓存逐条 MCP 调用 | 461/461 通过，`full_cache_validation.json` |
| 两实验并发同名栅格输出 | 分别在各自目录生成，像素数 14148 与 1800，`isolation_cache_validation.json` |
| 248 条官方公开 Kimi 轨迹的四个评分函数对照 | 同 GT、同路径归一化条件下，差异为零，`metric_formula_comparison.json` |
| 公开轨迹转标准消息后提取 | 保留全部 2,229 次调用 |
| GT 输出路径整体迁移 | 248/248 的参数分数仍为 1 |
| 实际 LangGraph 离线消息回放 | stream、答案、chat 与 usage 记录通过；反思两次格式尝试均计费，`actor_replay/validation.json` |
| 仅评第三轮的完整 pipeline | 保留未重试题目此前答案、累计调用和继承成本，`multiround_validation.json` |

全量回放通过证明当前修复版 GT 的路径、参数和工具调用可执行；脚本未将所有实际数值结果与科学真值逐项比较，不能据此宣称全部数值或空间分布正确。

多轮检查使用公开轨迹和明确设定的人工 token 数验证算术；它们不是实测模型成本或新的实验分数。没有新增测试文件。

## 运行前仍需处理

- 提供相同主模型、完整角色配置及预算；教师/压缩模型也必须纳入比较。当前 300 秒/题与递归上限不等于统一的总 token 硬预算。
- 本机缺少默认 HKG 的 `all-mpnet-base-v2` 权重；不能直接拿 MiniLM 当作同一个模型。尚未运行完整在线 baseline/HKG/FAR。
- 当前 `pyproject.toml`/`uv.lock` 没有声明完整运行依赖。现有环境按 requirements 安装；通过 `uv run --no-sync` 使用，`uv pip check` 已通过。不要把单独 `uv sync` 当作环境已准备完成。
- 新结果使用结构化 schema，旧日志与旧反思成本不做猜测或自动兼容。尚无完整角色成本记录的旧实验，不能补记为零。

工具缓存的合并与重建入口见 [tools/README.md](tools/README.md)，数据布局见 [benchmark/README.md](benchmark/README.md)。
