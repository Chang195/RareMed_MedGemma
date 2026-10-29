# 原流程 v2：历史排除、分页、BioC ID 兼容

本版本替代旧 run_original_test_pipeline.py 的新采集运行；旧运行完整保留。
原始 RareMed-Pairs 文件夹不修改，工作副本为 original-pipeline-new-test-v2。

## 变更边界

1. 保留历史 995 篇排除。
2. 同一检索式、每页 1000、同一默认排序，第一请求 URL 与原版完全一致。
   后续仅增加 cursorMark，直到无下一页。总命中不足 2000 时不虚构/补足。
3. BioC document id 为 unknown 时，允许从 article-id_pmcid / article-id_pmcaid
   获取 ID；所有非空 PMC ID 必须一致且等于请求 ID。冲突仍拒绝。

仅 collect.py 和 audit_licenses.py 变更。后者读取所有分页元数据，
不改变许可判断。其他 161 个文件、配置、全文筛选规则、采集规则、
DeepSeek prompt/schema/模型调用参数均与原版一致。
local_run/file_verification.json 和 *.diff 提供逐文件哈希与差异。
已复用校验通过的上一轮搜索第一页和三篇全文下载缓存，原字节不修改。

## VS Code 逐步运行

打开 RareMed_MedGemma，在终端运行：

```powershell
.\.venv\Scripts\python.exe run_original_test_pipeline_v2.py check
.\.venv\Scripts\python.exe run_original_test_pipeline_v2.py search
```

查看输出 searched_articles、historical_removed、new_articles 和
historical_overlap_after_filter（应为 0）。
若 new_articles=0，停止，不自动改检索式或新增疾病。

```powershell
.\.venv\Scripts\python.exe run_original_test_pipeline_v2.py collect-plan
.\.venv\Scripts\python.exe run_original_test_pipeline_v2.py collect
.\.venv\Scripts\python.exe run_original_test_pipeline_v2.py extract-plan
```

collect 原样执行全文筛选、MRI 图注采集和模型输入准备，不是付费调用。
extract-plan 仍只预览，真实 extract 必须显式 --allow-api，使用本地配置的密钥。
不要用旧 wrapper 或直接 pipeline.py search 绕过历史排除入口。
本次代码准备没有联网运行新搜索或调用 DeepSeek。

结果位于 original-pipeline-new-test-v2：
- local_run/history_filter_receipt.json：搜索后历史排除计数与哈希。
- reports/collection_v3_summary.json：全文筛选和图像采集结果。
- data/processed/collection_v3_figures.jsonl：原格式候选。

候选不是临床金标准，也不是审核完成的独立病例测试集。
split/export-sft 是原训练划分逻辑，暂不执行，以免把新病例分入训练。

## 已完成验证

4 项新增离线验证和 4 项原采集单元测试通过：分页与第二页元数据、
三篇真实缓存全文解析、错误 ID/冲突 ID 拒绝、非目标文件一致性。
