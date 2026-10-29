# RareMed-Pairs

公开病例文献的医学图文数据整理与信息抽取评测流水线。主题为 CADASIL；公开来源评测按 GBM、Astrocytoma、CADASIL 分组，使用 **MultiCaRe** 与 **MedPix 2.0**。

本项目处理整幅图、原文图注、病例关联和作者报告诊断；没有把来源诊断当作独立阅片确诊，也没有把 MultiCaRe 的疾病提及当作确诊标签。这里的评测是**文本抽取／关系恢复评测**，不是 MedGemma 的影像诊断成绩。

## 先运行公开的离线复算

Python 3.11 或 3.12。建议使用独立数据处理环境；不要替换另一台训练机器上的环境。

```powershell
python -m pip install -r requirements.txt
python tools/run_tests.py
python tools/replay_public_results.py
python tools/release_project.py
```

以上步骤不需要 API、模型权重或病例原文。`results/` 保存最终混淆矩阵、分病种统计及不含病例 ID／原文的分组混淆计数。离线复算检验最终 Precision、Recall、F1 和 95% bootstrap 区间是否一致，不能替代对逐条来源标签的审核。

## 当前目录

```text
pipeline.py                 唯一推荐的阶段入口
scripts/                    保留的实现、必要内部依赖及测试
configs/                    运行配置、抽取提示词、schema
metadata/                   脚本注册表、来源校验和、模型 revision、原检索队列 ID
results/                    可公开的最终汇总结果及复算统计
tools/                      下载校验、测试、离线复算、代码发布打包
.github/workflows/          无需 API 的自动检查
```

本地另有 `data/`、`reports/`、`models/`、环境、缓存和训练实验数据；这些不是 GitHub 上传内容。旧版本、取消的独立测试采集和临时构图文件已归档到项目同级的 `RareMed-Pairs_local_archive_20261028/`，不再混作当前入口。

`python pipeline.py list` 查看阶段；`python pipeline.py <阶段> --dry-run` 查看命令，不执行。

## 1. 文献检索与图文采集

```powershell
python pipeline.py search
python pipeline.py collect-plan
python pipeline.py collect
python pipeline.py article-flow
```

最终配置是 `discovery_1000.json`、`fulltext_screening_v3.json`、`acquire_v3.json`。检索上限 1,000，原冻结队列为 995 篇；队列 ID 在 `metadata/collection_lock.json`。新下载的检索结果可能随时间变化，要检查队列差异，不能强称同一队列。

最终采集策略：保留综述；许可阶段排除 CC BY-NC-ND；疾病及 MRI 线索检查全文；图注筛选 MRI 候选。采集保留其他／未知许可记录不代表可以公开分发。原图、图注、BioC 偏移、来源及校验和保留；此阶段不自动判定疾病、不拆图、不作患者去重。

主要输出：`data/processed/collection_v3_figures.jsonl`、`data/model_inputs/`、`reports/collection_v3_summary.json`。原完成记录为 869 条 figure、867 条可解码图像记录；图像记录不是独立病例数。

## 2. 病例与作者诊断关联

```powershell
python pipeline.py extract-plan
python pipeline.py extract --allow-api
python pipeline.py package
```

`DEEPSEEK_API_KEY` 只在本地环境变量配置，不写入代码、聊天或 Git。`--allow-api` 明确启动付费调用；本次文件整理未调用 API。

发送给模型的是原文段落、图号及图注文本，未发送图像像素。结构与引文校验通过不证明临床语义正确。格式修复仅限能够对回原文的空白与允许的冗余字段，实际入口使用 `extraction_format.py`。

主要输出：`data/processed/deepseek_collection_v3_links.jsonl`、`data/releases/candidate-v0.4/`。前者的 582 条是模型关联提案，不是 582 个确诊病例。

## 3. 数据整理、去重、划分和 SFT 导出

```powershell
python pipeline.py diagnosis-candidates
python pipeline.py split
python pipeline.py export-sft
```

依次生成逐图诊断候选、全图重复关系及保守分组、诊断任务划分和 MedGemma 风格的 image + messages 数据。导出目的地是新的 `data/datasets/diagnosis-sft-rebuild/`，拒绝覆盖已有目录。

本地冻结导出为 `diagnosis-sft-v0.2`（239 train / 24 validation / 33 test）。这些是图像记录，不是患者数；旧的粗疾病分组不是临床金标准。仅文件／像素精确重复作为确定同图；感知相似候选保守同组，不等于确认同患者。未知的跨文章患者重复及预训练污染不能被排除。

三组 20/50/80% CADASIL 混合实验另存于 `RareMed_CADASIL_SFT_Mixture_Experiments/`，不改变上述历史导出；公共代码包仅保留它的构建与核验代码，本地完整数据和冻结 inputs 不上传。它的严格二分类标签与旧粗分组的校正见该实验的本地审计报告。已取消的新独立测试集不属于最终项目。

### 可选：MedCPT 证据召回

```powershell
python -m pip install -r requirements-medcpt.txt
python tools/bootstrap_sources.py --seed-model-lock
python scripts/medcpt_local.py download
python pipeline.py retrieve
```

先按下节下载／核验来源文件，再运行 seed-model-lock。MedCPT 版本固定在 `metadata/medcpt_model_lock.json`。最终检索实现是 `medcpt_batch_v2.py`：多查询＋句子规则筛选＋召回／重排序，返回原文候选，不生成临床事实。`medcpt_batch.py` 是它仍在使用的内部依赖，不是另一套推荐入口。

本地已有前 10 篇的 v2.1 结果；全量 `retrieve` 是复现入口，不宣称之前已对全部文章完成该步骤。检索分数不是证据正确率；CASE 章节也不等于单患者范围已验证。

## 4. MultiCaRe / MedPix 三病分任务评测

先下载固定版本来源元数据，约 269 MB（含离线流程图资源）；不下载整库图像，不调用模型：

```powershell
python tools/bootstrap_sources.py --download
python tools/bootstrap_sources.py
python pipeline.py eval-scan
python pipeline.py eval-prepare
python pipeline.py eval-infer --allow-api
python pipeline.py eval-score
python pipeline.py eval-plots
python pipeline.py eval-ci
```

MedPix 固定作者仓库 commit；MultiCaRe 固定 Zenodo 2.1 记录；所有下载校验 SHA-256。配置和 URL 见 `metadata/source_lock.json`。

|任务|来源|实际测量含义|
|---|---|---|
|图像—病例关联|MultiCaRe、MedPix|原始图像／图注关联哪个来源病例；不是逐像素诊断|
|逐图原文描述|MultiCaRe、MedPix|原文图注恢复；MedPix 部分是结构化导入完整性|
|作者诊断名称抽取|MedPix|最终诊断字段抽取及分病种一对其余；没有某病阳性时不伪造敏感度|
|头部 MRI 来源筛选|MultiCaRe|图注规则对照 mri+head 来源标签；HEAD 比 brain 宽，不是视觉脑 MRI 识别|

分病种报告的主任务展示前三项，MRI 不加入该三病柱状比较。原始四项矩阵仍作为完整评测审计保留。

MedPix 的 History / Exam / Findings 原字段抽取另有最终入口：

```powershell
python pipeline.py medpix-prepare
python pipeline.py medpix-infer --allow-api
python pipeline.py medpix-score
```

详细方法见 [三病任务定义](docs/requested_tasks_gbm_astro_cadasil.md) 与 [MedPix 字段抽取](docs/medpix_glioma_evaluation.md)。缓存预测用于原始评测；在线模型版本和输出可能变化，重跑 API 不是逐字节重现旧输出的保证。

## 保留文件与内部依赖

`metadata/script_registry.json` 列出每个保留脚本的角色、依赖和 SHA-256。`four_disease_eval.py`、`global_three_disease_eval.py`、`task_benchmark_suite.py`、`evaluate_curation.py` 等仍提供最终评测需要的函数；其旧 CLI 不作为当前入口。不要只按名字删除依赖。

Orphadata 知识子图是独立的可选模块：[说明](docs/orphadata_knowledge_graph.md)。它不参与上述评测，也没有生成本次 SFT 数据。

## GitHub 上传与复现边界

```powershell
python tools/release_project.py --build
```

该命令生成同级的 `RareMed-Pairs_GitHub_ready.zip`，只包含允许的代码、配置、说明和汇总结果。**不要上传整个本地工作文件夹。** 本地原始数据、API 缓存、病例原文、模型和训练图像由 `.gitignore` 排除。

公共汇总可以离线精确复算；完整逐条历史评测还需要本地未分发的固定原文／响应缓存。重新检索和 API 推理可能受来源更新影响。代码授权和源数据发布范围见 [发布说明](docs/RELEASE.md)；本次未代为选择开源许可证或重新许可医学材料。
