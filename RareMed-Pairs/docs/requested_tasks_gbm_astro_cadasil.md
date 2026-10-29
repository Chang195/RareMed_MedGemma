# 胶质母细胞瘤、星形细胞瘤、CADASIL 全量来源检测

## 范围

按用户要求，仅使用MedPix和MultiCaRe，主评测不限制星形细胞瘤亚型。
GBM、毛细胞型、室管膜下巨细胞型、间变型等保留原始来源名称；明确GBM优先映射到GBM，不能同时算星形细胞瘤。
疑似、未知诊断不自动作为确诊答案。该映射复用来源命名，不将历史标签重新解释为WHO 2021分子分类。

1. 病例叙述：实际测图片—病例叙述归属，使用两库来源关联，不声称测量全部临床字段抽取。
2. 逐图描述：MultiCaRe以独立BioC原文解析结果对照来源图注；MedPix运行现有结构化适配器，仅计导入完整性，不能称为独立语义配对性能。
3. 诊断标签：仅MedPix，使用其最终诊断字段，三目标类别加Other、Unresolved。
4. MRI筛选：MultiCaRe全量130,791图像/子图记录。来源标签只有mri+head；HEAD不是精确brain标签。本任务准确名称为“头部MRI来源标签筛选”，预测器仍是原有图注MRI规则。

没有每病、每文档、每图的抽样上限。病例任务扫描所有本地来源病例，保留所有命中文档中的病例和来源关联图；8幅图一请求仅是批处理。
MRI任务不按病种先过滤，否则会丢失大量重要负例。
MultiCaRe没有统一最终诊断字段，按文档疾病提及分层；MedPix按最终诊断分层，不混用两种定义。

## 运行

VS Code终端进入RareMed-Pairs；当前终端已配置DEEPSEEK_API_KEY后：

```powershell
python scripts/evaluate_requested_tasks.py run --fetch-text --allow-api
```

--fetch-text允许下载尚未缓存的公开BioC文本，不下载原图；--allow-api允许付费文本抽取。
不带--allow-api只做离线准备/已有原文解析和请求规模预览，不调用服务。
新基准输入与旧CADASIL/MS/NMOSD或小样本测试不同，不可直接复用旧成绩。

可以分步：

```powershell
python scripts/evaluate_requested_tasks.py prepare --fetch-text
python scripts/evaluate_requested_tasks.py infer --allow-api
python scripts/evaluate_requested_tasks.py score
python scripts/plot_requested_task_matrices.py
```

中断后重跑run命令：推理会跳过已有请求记录，不自动重复付费；失败请求保留原始状态。
缺全文、超长上下文、来源图注冲突等记录单独写入held文件，不静默剔除后宣称全量成功。

## 看报告

```powershell
Invoke-Item .\reports\gbm_astro_cadasil_v1\report.html
Invoke-Item .\reports\gbm_astro_cadasil_v1\plots.html
```

图表脚本输出独立PNG与SVG：reports/gbm_astro_cadasil_v1/plots/。

## 混淆矩阵定义

二分类统一为行=来源参考、列=预测：[[TN,FP],[FN,TP]]。

- 病例关联：每个“图像—候选病例”关系为一个二分类对象，候选病例为该次输入中的全部病例；同时报告图数/独立来源病例数。单病例文档没有负关系，无法估计其特异度。多病例拒答不叫错误诊断，但对应来源关系未检出。
- 原文描述：每个“图像—候选原文图注”关系为对象，候选为同文档不同来源图注，以及预测新增文本。比较规范化文字一致性；同时输出Word-F1和ROUGE-L，不能只靠矩阵解释文字质量。
- MRI筛选：所有来源图像/子图为对象，mri+head为正例，其余为负例；另画只含MRI的部位区分矩阵。标签属来源分类，未新增独立专家逐图审核。
- 诊断：5×5多分类矩阵（GBM、Astrocytoma、CADASIL、Other、Unresolved），另有每病一对其余2×2矩阵。主评测不限星形细胞瘤亚型。

没有可评分记录时图会标明NO SCORED RECORDS，不伪造零分通过。推理尚未完成时仅用通过格式/引文校验的输出计算主矩阵；覆盖率、未运行和无效输出数量同时展示。病例关联来源答案自动整理，不等于独立专家诊断金标准。

同一病例多图不独立；图像级指标不能代替病例级临床诊断性能。全库扫描范围仅限当前本地公开版本，不代表医学文献全部病例。
