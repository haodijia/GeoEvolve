# 固定感知观测与参考一致性修复

本版 `2026-10-31-contracts-v3` 将经审核的补充观测直接合并进数据集
`perception/observations.csv`，共 569 条。运行不再读取外置补丁，不需要模型服务。
旧补丁加载器、下载入口和版本锁已删除。

## 修复边界

保留全部 248 题的 `auto`、`instruct` 和 104 个工具的公开参数、类型注解、
description。修复工具实现、固定缓存与标答不一致，以及参考链的路径和数值传递。
被测 agent 的运行轨迹和历史成绩不修改，也不把其规划或传参错误修成正确调用。

- 栅格统计仅统计目标波段，排除 NoData 和非有限值；均值合成按有效像素计数。
- 归一化指数排除无效反射率和零分母，按地理网格配准。
- TES 不再把亮温差当成发射率差，使用简化 NEM / beta / MMD 估计；这不是完整的大气校正 TES 产品。
- ATI 应用反照率 scale / offset，保留 `(1-albedo)/(day-night)` 的符号，移除错误裁剪。
- 数据修订包括反照率 scale、热红外网格方向，以及从同源题补齐 Q179 的 26 个输入。栅格像素不做凑答案修改。
- 参考链实际执行后记录输入、输出；后续计算引用真实返回值。修复读错波段、月份、城市、合成输出、比例单位和 bbox 格式等问题。
- 按实际观测修正必要的选项数值、关系和答案字母；原版问题、参考与缓存归档在 `provenance/previous_release/`。

具体例子：Q206 篮球场计数修为 4、6、3；Q232 实际中心约为 (2654,1369)，答案改为 A；
Q204 计数为 3、4、4，Q205 冻结观测为 28、28、32，各自修正排序选项；Q227 修正西侧目标框。
Q205 的数值是发布缓存观测，不宣称人工重标全部目标。

本版共 129 题存在选项或答案修订：108 题改动选项内容，27 题改动答案字母或开放题答案，
两者有交集。完整改动保存在 `provenance/answer-corrections.json`，不覆盖上版归档。
Q96 修复的是数据集参考解法的统计对象，Q147 修复的是旧参考转换器对合成图路径的映射；
这些修改针对 bench 自带参考，不修改被测 agent 的实际调用。

## 补充观测来源

补充请求来自三轮、每轮 248 题，共 744 条历史轨迹。只提取请求，审核结果后冻结：

- 检测候选来自 YOLO11l-OBB / YOLO11x-OBB 共识，人工检查叠加图；不把共同未检出当可靠零值。
- 分类候选来自 RemoteCLIP ViT-L/14 与 GeoRSCLIP RS5M ViT-B/32。
- 掩膜来自 ChangeOS ResNet101、SAM2.1 Hiera Large；旧 Q214 掩膜来源仍为 SAM2.1 Hiera Small。
- 由完整检测列表推导同条件计数或唯一目标框时，保留来源，不声称运行过原 InstructSAM / RemoteSAM / Strip-R-CNN。
- 少量明确同义表达通过双向 NLI 和人工审核后作为固定别名；运行时不加载 NLI，也不做任意语义近邻检索。

来源和修订见数据集 `provenance/observations-v3.json`。原始服务实验的 SQLite 输出不参与评测。

## 缓存边界

键包含工具、图像顺序、规范化 prompt 和完整 bbox。规范化不删除方位、大小、颜色等条件；
相同图像仅通过逐字节比对共享观测，不新增文件哈希。未知请求明确报错，不以零或猜测代替。
缓存不向 agent 提供每题词表、答案或匹配提示。

## 验证

```bash
uv run --locked python -m benchmark.prepare
uv run --locked python scripts/validate_mcp_tools.py --replay-ground-truth
uv run --locked python -m benchmark.rebuild_references --work-dir /path/to/new-reference-replay
```

验证器通过 MCP 接口逐次核对返回值、全部固定观测和 104 个工具调用。
`provenance/answer-review.json` 保存从返回值到答案的算术证据；`answer-corrections.json` 保存选项与答案改动。
这些维护材料不进入 agent 提示词。Q96 的分位数由真实每日均值按线性插值计算；
部分源题的日期文字与提供文件日期不一致，保留题干并在证据中记录实际日期。

本版核验通过 248 条参考链、2341 次 MCP 返回值比较、248 题答案算术证据、
569 条缓存调用和 104 个工具检查；公开接口与题干逐项对比无改动。
发布内的紧凑报告位于 `provenance/validation-v3.json`。这表示参考执行一致，不是模型准确率。
