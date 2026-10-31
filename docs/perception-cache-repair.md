# 三轮轨迹感知缓存修复

本机补丁版本 `trajectory-repair-v1`，保存于
`/home/dijkstra/data/earthbench_perception_patch/observations.json`。
模型和审核材料保存在数据目录；未修改发布数据集，未重跑 LLM 全量评测。

补丁现已发布到 ModelScope 数据集 `haodijia/EarthBench-Fixed` 的
`perception/patches/trajectory-repair-v1/`，固定 commit 为
`2e260612848b3e2b1c55367f82e4d84c01ccbb94`。原数据版本保持不变；新增观测与掩膜
作为独立补丁安装在数据集旁。代码侧 `benchmark/earthbench_perception_patch.lock.json`
固定发布位置与版本，不新增文件哈希。

```bash
uv run --locked python -m benchmark.prepare_perception_patch
uv run --locked python -m benchmark.prepare_perception_patch --check
```

## 范围和覆盖

来源为以下三轮，各 248 题，共 744 条轨迹：

- `earthbench_qwen36_27b_vllm020_native_20261029_212157`
- `earthbench_qwen35_9b_vllm020_native_20261030_100835`
- `earthbench_qwen36_27b_live_perception_20261030_220450`

仅提取工具调用的输入，从原图生成观测，未从题目选项、标准答案或是否答对推导结果。
去重后有 1,096 种有效请求，另有 9 次路径或参数结构无效的调用。
这里的覆盖率是历史请求命中率，不是任务正确率，也不是感知模型准确率。

| 工具 | 有效请求种类 | 原缓存命中 | 修复后命中 |
| --- | ---: | ---: | ---: |
| MSCN | 236 | 172 | 184 |
| RemoteCLIP | 207 | 148 | 161 |
| SM3Det | 184 | 36 | 69 |
| Strip-R-CNN | 109 | 0 | 4 |
| InstructSAM | 215 | 48 | 81 |
| RemoteSAM | 110 | 0 | 3 |
| SAM2 | 8 | 2 | 6 |
| ChangeOS | 27 | 16 | 25 |
| 合计 | 1,096 | 422（38.50%） | 533（48.63%） |

原缓存统计包含通用表达规范化和相同图像字节的路径扩展。
补丁包含 109 条去重观测：7 条发布观测的同图路径扩展、58 条经审核的模型共识结果、
23 条由发布检测列表明确推导的计数或框、18 条原分割模型输出及其同图扩展、
3 条审定语义别名。不同原始表达可以使用同一规范化观测，因此新增命中请求数为 111。
共审核 116 条候选，拒绝其中 12 条；同图扩展和规范化去重后得到最终补丁。

## 如何补充

- 检测：YOLO11l-OBB 与 YOLO11x-OBB，均为 DOTA 模型。采用 0.35 阈值、1024 推理；
  大图追加原分辨率切片。只有数量相同且框可一一对应、IoU 至少 0.5 的正检出才成为候选，
  再检查原图叠加结果。不把双模型均未检出解释为可靠零计数。
- 分类：RemoteCLIP ViT-L/14 与 GeoRSCLIP RS5M ViT-B/32，top1 一致且双方第一、第二名
  概率差至少 0.1，再检查原图。剔除了棒球场→公园、网球场→足球场及不可靠的混合地类。
- 掩膜：ChangeOS ResNet101 与 SAM2.1 Hiera Large。使用实际图像顺序和原请求框，
  不替 agent 改框；剔除建筑分割把网球场、港口误识别的结果及大面积背景掩膜。
- 确定性推导：发布 SM3Det 的完整列表可以生成同条件计数、唯一对象框和舰船检测列表。
  保留来源及推导操作，不声称运行过 InstructSAM、RemoteSAM、Strip-R-CNN 原论文模型。
- 语义别名：同工具、同图、同框，先核对类别和限定条件，再以 DeBERTa NLI 双向蕴含
  至少 0.98 筛选，最后审核。本轮只接受 3 个具体图片下的 `tennis`→`tennis court`。
  运行时不加载 NLI，不进行任意近邻检索，也不记录每次命中日志。

两个检测模型属于同一模型家族，共识不能证明结果正确。分类 softmax 数值也不是经过
校准的正确率。新模型输出保留实际后端与审核记录，不视为人工真值或全部原工具权重。

下载模型位于 `/home/dijkstra/model/earthbench-cache-repair/`：两个 YOLO-OBB 权重、
RemoteCLIP ViT-L/14 和 `cross-encoder/nli-deberta-v3-small`。
复用了 `/home/dijkstra/model/GeoRSCLIP/RS5M_ViT-B-32.pt`、
`/home/dijkstra/model/ChangeOS/changeos_r101.pt` 和
`/home/dijkstra/model/sam2.1_hiera_large/sam2.1_hiera_large.pt`。

## 匹配边界

缓存键包含工具、图像及顺序、规范化提示词和完整 bbox。通用规范化处理大小写、
分隔符、单复数和已有类别同义词；方位、颜色、大小等条件保留。不同图片只在输入文件
逐字节相同时共享记录，不生成文件哈希。不改变 agent 可见提示词或给它指定本题词表。

原有 461 条观测全部保留。补丁遇到同键不同结果直接报错；内容完全相同、保存路径不同
的掩膜允许视为等价。普通足球场请求不会命中带“最西侧”条件的框。
RemoteSAM 面对多个对象时不擅自选择；Strip-R-CNN 只补舰船请求。
空提示词不能借用该图片下其他类别的唯一记录。

仍有 563 种请求缺少可靠结果，包括 105 种 Strip-R-CNN 非舰船请求、107 种缺少唯一定位
或完整条件的 RemoteSAM 请求、98 种分类未通过筛选的请求，以及其他检测类别、条件、
模型分歧、未检出或审核拒绝的请求。它们继续明确报错，不返回猜测结果。

## 使用和复现

评测仅使用离线缓存，已移除服务请求、服务探测、HTTP 后端和启动脚本。
默认读取数据集真实目录同级的
`earthbench_perception_patch/observations.json`，也可用
`EARTH_BENCH_PERCEPTION_PATCH` 指定 JSON。原始实时服务 SQLite 库不再用于离线命中，
避免复用此前不可靠的 DINO 结果。新增观测由离线脚本生成，经过审核后冻结。
实验的 `perception_cache.json`、`run.json` 与 LangSmith 元数据记录补丁路径、版本和条数。

补丁数据与审核中间材料：`/home/dijkstra/data/GeoEvolve/cache-repair/v1/`。
其中 `coverage.json` 为实际离线查询统计，`review.json` 为全部候选审核决定，
`unresolved.json` 为仍未解决的请求，`unresolved_summary.json` 为分类汇总。

```bash
# 主项目环境执行审计、冻结；生成模型候选使用下面的独立环境。
uv run --locked python scripts/repair_perception_cache.py audit \
  --directory /home/dijkstra/data/GeoEvolve/cache-repair/v1
uv run --locked python scripts/repair_perception_cache.py freeze \
  --directory /home/dijkstra/data/GeoEvolve/cache-repair/v1 \
  --destination /home/dijkstra/data/earthbench_perception_patch

cd scripts/perception_cache
source ./env.local.sh
# COMMAND 可选 detections、classifications、masks、aliases。
uv run --locked --no-sync python ../repair_perception_cache.py COMMAND \
  --directory /home/dijkstra/data/GeoEvolve/cache-repair/v1
```

重新生成候选后必须重新审核 `review.json`，再冻结。不要将候选直接当作正式补丁。
本轮验证了所有 461 条原观测结果、全部历史有效请求、限定条件不串用、bbox 严格匹配、
灾前灾后顺序保留，以及掩膜在题目输出清理后仍可再次复制。未编写新增测试，未重新评测
LLM 任务正确率，也未同步或启动 A800 实验。
