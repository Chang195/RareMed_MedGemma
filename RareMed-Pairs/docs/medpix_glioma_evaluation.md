# MedPix：胶质母细胞瘤 vs 星形细胞瘤

本轮仅用MedPix 2.0，测试文本抽取，不测试看图诊断。

## 运行

在RareMed-Pairs项目终端运行：

```powershell
python scripts/medpix_glioma_eval.py run --allow-api
```

需要当前终端已设置DEEPSEEK_API_KEY；不要发送到聊天中。此命令准备缺失的输入，调用5个DeepSeek请求，每完成一组自动评分。已存在的请求记录不会重复收费调用；中断后运行同一命令继续。已有失败请求也不自动重试，避免未知计费下重复请求。

不调用API的检查：

```powershell
python scripts/medpix_glioma_eval.py run
```

重新评分与打开报告：

```powershell
python scripts/medpix_glioma_eval.py score
Invoke-Item .\data\benchmarks\medpix_glioma_v1\report.html
```

## 固定范围

- 遍历全部671个病例、2050张图，不设抽样上限。
- 从原始Case Diagnosis字段筛选，而不是从病例全文疾病提及中筛选。
- 胶质母细胞瘤5例18图，明确的非胶质母细胞瘤星形细胞瘤9例19图；包括毛细胞型、室管膜下巨细胞型、促纤维增生婴儿型等亚型，不按分级排除。
- 1例原字段为“Pontine Glioma probably astrocytoma”单列held.jsonl，不自动提升为确诊。
- 保留这些病例的全部对应图，包含来源CT/MR等；本轮未额外限定“Brain and Neuro”标签或MRI模态。模态与部位保留在逐图参考中。
- 全部14病例37图，5组，每组3个候选病例；末组补1个无关病例作为干扰，不把其图或病例计入主评测样本。
- 使用匿名病例标签和打乱的图ID；输入没有来源U_id、图像文件名或病例—图片答案映射。

## 四项任务

1. 病例叙述：提取History、Exam、Findings完整原文，检查每字段的文字精确匹配、Word-F1、ROUGE-L，以及缺失字段是否返回null。各字段有多少非空参考单独报告，不把空字段算作文本召回。
2. 图片—病例关联：从三个候选病例中选来源病例，报告关系TP/FP/FN及P/R/F1。源文本不能明确关联时允许null，按未检出关联计分。
3. 诊断抽取：提取原始Case Diagnosis完整文字，报告文字指标；另报告来源诊断两类标签匹配。该任务允许读取原文中的诊断，不是让模型从症状或像素重新诊断。
4. 逐图描述：从来源图注段落复制完整图注，比较独立保存的原始Caption字段。候选图列表仅含ID，模型需要从来源段落取出对应图注。

本版本为原文字关联任务增加了病例字段和图注原文输出，提示词与输出结构有版本变更，不与以前原始提示词结果合并。它检验结构化来源适配后的文本抽取，不能宣称同样成绩代表PMC复杂全文解析质量。

## 结果与核查

输出目录：data/benchmarks/medpix_glioma_v1。

- report.html、summary.json：按疾病、任务分层结果及状态。
- case_results.jsonl：逐病例、逐叙述字段的原文和预测。
- figure_results.jsonl：逐图来源病例、诊断、图注和预测。
- case_references.jsonl、figure_references.jsonl：独立评分文件，不发送给模型。
- held.jsonl：不确定诊断记录。
- inputs：匿名模型输入与实际提示词。
- outputs：原始API结果、验证错误、token用量和收据。

来源、脚本、提示词与输入均冻结哈希。格式或原文引用校验不通过时整组记为无效；无效和未运行输出保留在计划分母中，不能把剩余有效结果冒充全量成功。所有请求尝试完毕后complete=true；仍需检查validation_failed数量。

同病例多图相关，病例数不等于图像数；不据37张图声称大规模临床验证。没有新增人工专家真值。
