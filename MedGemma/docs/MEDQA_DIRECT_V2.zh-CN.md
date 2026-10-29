# MedQA 直接回答协议 v2

当前状态（2026-10-28）：按用户要求恢复本协议；长推理官方配方任务已停止，数据保留。复用已完成Base与A的部分记录，继续A/B/C，不更改冻结prompt或评分。日志改为`resume-run.log`；GBM/astrocytoma队列恢复等待本协议成功结束。

旧协议只比较第一个位置的字母概率，未验证该位置是否为答案，已发现分析/格式token竞争。本版本不覆盖旧预测、PPT或诊断成绩。

## 新方法

固定英文指令：

```
Select the single best answer to this medical multiple-choice question. Return exactly one uppercase letter from A, B, C, D. Do not include reasoning, explanations, markdown, punctuation, or any other text.
```

之后接原始题目和原始顺序的A/B/C/D选项。不添加assistant答案前缀，不强制屏蔽其他token，不要求模型输出长推理。

实际调用greedy generate，max_new_tokens=32。只删除已声明的末尾EOS token和首尾空白，严格接受一个大写A/B/C/D。其余回答、解释、截断都单列为格式失败，并在总准确率中算错，不擅自截取第一个字母。raw_output存真实生成，不存评分器伪造字母。

更强prompt可以提高格式遵守率，但不能保证所有题都符合格式，更不能保证准确率复现官方。若应用必须只展示一个字符，可以由程序仅返回成功解析的prediction；失败应返回明确错误/abstention，而不是编造字母。硬约束输出虽然能保证字母格式，仍需独立验证准确率。

## 先验证再测试

`medqa-direct-v2/protocol.json`在推理前固定指令、生成参数、源码哈希及32个训练划分样本。seed20261028，排除之前8个排查样本及与测试集题目完全相同的记录。

Base/A/B/C分别验证32题。预先固定的门槛为每个模型至少29/32题得到完整、严格单字母输出；门槛只看格式，不依据正确率选择prompt。训练划分验证可能是原模型见过的数据，只作格式检查，不能报告为泛化成绩。

通过后才允许统一重跑原来1,273道MedQA测试题。旧测试结果已查看，因此新协议属于事后修复后的重新评测，不能声称完全预注册或未接触过测试集。新指令没有按测试题的分数调优。本协议仍不是官方逐步生成协议的完整复现。

## VS Code 查看与恢复

后台运行时不要重复启动：

```powershell
python run_medqa_direct_v2.py status
Get-Content .\medqa-direct-v2\resume-run.log -Tail 20 -Wait
```

确认原worker已退出后，可恢复：

```powershell
python run_medqa_direct_v2.py run
```

逐题原子写入，匹配冻结协议的已保存结果会跳过。Windows临时文件访问冲突有重试。模型、环境、量化与adapter选择沿用原实验；不上传、不训练、不下载权重。

输出在`medqa-direct-v2/`：protocol.json、validation-report.json、predictions/validation与predictions/test、summary_metrics.csv、medqa_item_results.csv、paired_report.json。完整测试完成后计算Wilson95%准确率CI、相对Base的配对题目bootstrap差值CI，10,000次、seed42，并报告格式失败/截断。相似成绩不等于证明其他能力不变坏。
