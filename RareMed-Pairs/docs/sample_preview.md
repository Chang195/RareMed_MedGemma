# 逐图样本预览

在 RareMed-Pairs 的 VS Code 终端运行：

```powershell
python scripts/preview_dataset.py
Invoke-Item ".\reports\sample_preview_v1\preview.html"
```

需要 Python 3.10+ 和 Pillow，无需模型、GPU或API。
默认读取 `data/processed/medcpt_pilot10_v2_1/evidence_records.jsonl`。
使用 `--source` 和 `--output` 可为后续批次生成独立预览。

输出目录 `reports/sample_preview_v1` 包含：

- `preview.html`：自包含的本地页面，内嵌图片，按逐图记录展示原始图注、任务问题、目标候选和原文证据。
  顶部可筛选作者报告确诊的单病例候选、诊断不确定候选、其他暂存记录。
- `samples.jsonl`：逐图机器可读草稿，保留图像、原文位置、临床候选、诊断依据和任务状态。
- `samples.csv`：逐图概览。
- `summary.json`：数量和输入/输出校验值。

脚本检查输入清单、候选池、文章原文和图片哈希，并逐字核对证据片段。
图片在页面中转换为JPEG用于展示；原始图像文件不修改，原路径与哈希保留。
该页面包含完整病例证据，仅供本地查看，不自动上传或发布。

这是样本设计草稿，不是SFT导出：

1. 图像描述任务展示完整原图注作为目标候选，不自动删改图注中的诊断、非视觉信息或面板标签。
2. 诊断任务仅为作者报告确诊且未命中非患者/动物/关联不明文字信号的记录展示目标候选。
   “未命中”不表示已验证为人类病例，作者确诊也不表示可仅凭图片推断。
3. 临床文本仅保留为文章级候选，`clinical_context_assigned` 和任务的 `input_clinical_text` 均为空。
   不把文章检索结果自动分配给同文章每幅图。
4. 多病例、统计图、动物模型等通过文字启发式暂存，原数据不删除；这些规则会有误判与漏判。
5. `training_ready` 始终为 false，`sft_exported` 为 false。没有划分训练测试、近似去重、完整脱敏或独立审核。

原图注、临床上下文候选和诊断证据明确区分；图注不放入描述任务的用户输入，
诊断依据也不自动放入疾病分类任务的输入，避免直接抄答案。
