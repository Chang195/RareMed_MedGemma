# MedQA 后自动评测 GBM 与星形细胞瘤

当前依赖（2026-10-28）：用户已取消耗时过长的官方风格长推理，恢复直接回答v2。本队列等待`medqa-direct-v2`，不再等待官方风格任务。旧队列日志均保留，当前日志为`queue-direct-resumed.log`。

已准备依赖队列。它只读取当前 MedQA 的完成记录，不与其同时加载GPU模型。等 `medqa-direct-v2` 的四个模型各完成1,273题、配对报告生成且worker退出，才串行启动新的影像任务；父任务失败/停止时队列停止并报告，避免误认为已完成。

## 固定任务定义

这两项是“图片→来源作者诊断标签”的探索性二分类，不是新增SFT或临床金标准验证。

| 任务 | 来源阳性 | 其他来源诊断 | 总图片数 | 阳性来源分组 |
|---|---:|---:|---:|---:|
| GBM vs其他来源诊断 | 8 | 58 | 66 | 2 |
| 来源标为astrocytoma vs其他来源诊断 | 2 | 58 | 60 | 1 |

GBM阳性来自MPX1134、MPX1184；astrocytoma阳性来自MPX1694，原文标签为`Recurrent high-grade astrocytoma`。保留这些作者标签，不据此推断IDH状态或按现代WHO分类重写诊断。

为避免GBM/astrocytoma分类关系造成不可靠阴性，GBM任务排除2张来源astrocytoma图；astrocytoma任务排除8张来源GBM图。具体纳入/排除ID、原始诊断及原因均保存在`protocol.json`，不修改原测试集。两个任务用同一组58张其他来源诊断图作为对照；这58张并非独立临床审核的阴性。各任务四个模型的测试集完全相同。

输入只含原图RGB和新的冻结user问题，不包含原图注、病史、metadata、来源诊断或assistant参考答案。图片原文件及箭头不改动。沿用原模型revision、处理器、NF4量化和原先选定的A25/B25/C75；Base不加载adapter。不额外训练。

本次读取来源标签和抽查阳性图不能构成独立临床审核，也未将全部图重新审核为序列匹配的MRI专属队列；因此称为来源影像标签评测。只有一个astrocytoma阳性分组，CI可能退化，不能用结果证明其他能力不变坏。测试图此前已用于CADASIL任务，不是全新独立队列。

## 实际执行

模型顺序Base→A→B→C。每个模型依次做GBM和astrocytoma任务。各任务先取既有train中的一个阳性和一个其他来源诊断图做格式/加载检查，失败即停止，不根据测试输出修改prompt。之后推理固定test子集。

Greedy生成，最多32tokens，使用相同处理器。按已固定的明确标签/同义词规则解析；已声明EOS从token序列末尾删除，避免此前终止token解析问题。截断、仅提及、不确定和无法解析均保留，不简单用包含GBM/astrocytoma判断阳性，也不强制阴性。存在未解析回答时，主要二分类估计暂不报告。

全部完成后自动报告TP/FN/FP/TN、Recall、FPR、specificity、precision、F1和balanced accuracy、无法解析数量及逐样本改善/退步。CI按保守来源分组做10,000次class-stratified bootstrap，seed42，不把多图当独立患者，也不把零宽区间当作没有未来错误。astrocytoma仅1个阳性来源分组，无法估计阳性组间不确定性：Recall及依赖它的BA/precision/F1置信区间标记不可估计；保留点估计及分母，不绘制误导性的零宽区间。

## VS Code 查看

队列已在后台等待时，不要再次启动：

```powershell
python queue_tumor_after_medqa.py status
Get-Content .\tumor-binary-evaluation-v1\queue-direct-resumed.log -Tail 20 -Wait
```

结束日志跟踪的Ctrl+C不会停止独立后台队列。保持接电、不关机/休眠；无需保持VS Code窗口打开。

如电脑重启或队列因父任务失败停止，先完成/恢复MedQA，再运行：

```powershell
python queue_tumor_after_medqa.py run
```

脚本有队列锁并校验冻结源码；不要同时手动运行底层GPU任务。逐条预测可恢复。队列不会修改当前MedQA评测代码或旧结果。

## 输出

全部保存在当前项目的`tumor-binary-evaluation-v1/`：

- protocol.json：标签定义、模型/源码/图片哈希、纳入/排除清单、推理与解析规则。
- queue-config.json、queue-status.json、queue-final.log、queue-final-error.log：依赖与进度。
- probe-{model}-{task}.json：训练图预检。
- predictions/{task}/{model}/*.json：逐图原始输出、输入IDs、来源标签和分组，可断点恢复。
- summary_metrics.csv、per_sample_results.csv、paired_changes.csv、report.json：统计及配对变化。

预计在MedQA之后额外需要约25–35分钟，实际取决于生成长度。若预检失败，队列会保留错误并停止，不悄悄改动实验定义。
