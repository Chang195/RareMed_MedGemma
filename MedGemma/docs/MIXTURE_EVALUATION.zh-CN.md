# Base / A / B / C：诊断与完整 MedQA 评测

此脚本只进行本地推理，不训练、不改权重、不上传、不下载，也不重装环境。

## 本次固定条件

- 原始模型与 revision、NF4 double quant / FP32 / SDPA、处理器沿用已验证配置。
- A/B/C 使用验证集选择的 step 25 / 25 / 75。Base 独立加载原始 snapshot。
- 诊断：同一验证集 67 张，然后同一测试集 68 张；输入只有原图 RGB 和已有 user 二分类问题，去掉 assistant 参考答案。生成最多 32 tokens、greedy decoding，四模型一致。
- 诊断解析只接受完整明确标签；不确定、多诊断、仅提及、无法解析和截断均保留，不自动算阴性。存在未解析回答时，绘图脚本暂不强制给出二分类成绩。
- MedQA：`GBaker/MedQA-USMLE-4-options`，revision `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`，test 1,273 题。复用已下载且已校验哈希的题集；每个模型相同 zero-shot prompt，以 A/B/C/D 下一 token 对数概率的最大值评分。不是自由生成答案或思维链评分，不能与其他评分协议混称同一 benchmark 数值。
- 所有输入 token ID、原始诊断输出、MedQA 四选项 log probabilities、checkpoint、协议哈希、耗时均逐条保存。
- 自动检查 MedQA 完整问题文本/连续 20 词与三组训练记录重合；这不是语义改写或预训练污染的排除证明。

## 在 VS Code 终端查看进度

在当前项目根目录运行（无需激活 .venv、无需 PowerShell 脚本权限）：

```powershell
python mixture_evaluate.py status
Get-Content .\mixture-evaluation-v1\resume-io.log -Tail 20 -Wait
```

日志窗口 `Ctrl+C` 只退出 `Get-Content` 跟踪，不会停止独立后台推理。不要再次启动第二个评测实例；脚本有 worker lock 防止重复占用显存。

## 中断后继续

先确认先前 worker 已退出，再运行同一命令：

```powershell
python resume_mixture_evaluation.py run
```

程序校验冻结协议、源码、数据及 adapter 哈希，跳过已完成并匹配协议的逐条 JSON 文件。不要手改源码、prompt、checkpoint 或已保存的结果来“修正”测试成绩。发生错误查看 `mixture-evaluation-v1/last-error.json` 和 `run-error.log` / `run.log`；保留错误记录。

首轮运行在 Base MedQA 保存 62 题后因 Windows 拒绝替换 `status.json` 而停止。`resume_mixture_evaluation.py` 仅替换文件写入函数：唯一临时文件、flush/fsync、遇 PermissionError 后有限重试；不修改冻结推理代码、协议、prompt、评分或已保存预测。恢复证据在 `io-recovery-*.json`，新日志为 `resume-io.log` / `resume-io-error.log`，原错误日志保留。后续请使用此恢复入口。

如需仅跑某个模型，可用 `--models Base` 或 `--models A B C`；全四模型齐全才导出完整比较和图表。不要用 partial 结果误称最终成绩。

## 输出

`mixture-evaluation-v1/` 位于当前 VS Code 项目内，只保存轻量评测记录。原始大模型与 adapter 仍从 G 盘读取，不复制、不覆盖。

- `protocol.json`：冻结协议、模型/题集/代码/处理器版本、哈希、环境与评分规则。
- `overlap-audit.json`：自动文本重合检查。
- `probe.json`：Base 和 A 的正负验证图与 QA 有限 logits / token 边界检查。
- `status.json`、`worker.lock`、`run.log`、`run-error.log`：进度与运行日志。
- `predictions/{Base,A,B,C}/{diagnosis-validation,diagnosis-test,medqa}/*.json`：原子写入的逐条结果，可恢复。
- `completed/*.json`：完整任务的完成记录。
- 全部结束后生成 `diagnostic_predictions.csv`、`medqa_item_results.csv`、`medqa_reference.csv` 和 `figure_input_manifest.json`。
- 自动运行已冻结的统计绘图代码，真实结果另存 `MedGemma_SFT_Figures/actual_results/`；原来的 illustrative 文件保留。此步骤输出 PDF/SVG/300dpi PNG/summary CSV，不会自动把待填充 PowerPoint 改成真实成绩。

## 电脑运行注意

连接电源，系统睡眠设为“从不”，不要关机、休眠或手动停止 Python。可以关闭 VS Code 界面和显示器；本次后台 worker 不依赖 VS Code 终端存活。避免同时训练/推理其他模型，或打开大量占显存的软件。后台运行不会阻止 Windows 更新重启；重启后用同一命令恢复。

## 限制

只有一套训练随机种子；测试 CADASIL 6 张 / 4 来源分组，患者身份未核实；部分 holdout 此前已查看，不能声称全新独立临床验证。正负标签与 RareMed / MedPix 来源相关；作者诊断不等于 MRI 可独立推出的临床金标准。MedQA 是知识保持检查，不能证明全部其他能力不变坏。诊断 CI 按保守来源分组 bootstrap，MedQA 精度 CI 用 Wilson，模型差值用题目配对 bootstrap；详见绘图 README。
