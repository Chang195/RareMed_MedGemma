# MedCPT 本地部署与使用

在 RareMed-Pairs 项目目录的 VS Code PowerShell 终端运行。无需激活环境：

```powershell
.\.venv-medcpt\Scripts\python.exe scripts/medcpt_local.py search --article PMC12675991 --section CASE --query "genetic testing evidence for CADASIL diagnosis" --top-k 3
```

支持英文查询，返回本地论文的原文片段、段落编号、原文位置、召回与重排序分数。
推理强制离线，正文不发送给 API。结果保存到 `reports/medcpt_PMC12675991.json`；
同一篇文章的下一次查询会覆盖这个演示结果文件，不修改原始数据集。
当前是按文章范围检索，不保证片段属于同一患者；分数不是概率或证据核验。
`--section CASE` 只保留源数据标记为 CASE 的病例章节，减少讨论、结论和背景干扰；
并非所有文章都有 CASE 标签，找不到时会报无文本，去掉参数即可检索全篇。
病例章节仍可能包含多个患者，不能将章节过滤当作患者身份核验。
这是部署与检索验证，尚未批量建立病例文本数据，也未测量准确率或 Recall@k。

## 组件

- `models/medcpt/query`：NCBI MedCPT Query Encoder。
- `models/medcpt/article`：NCBI MedCPT Article Encoder。
- `models/medcpt/cross`：NCBI MedCPT Cross Encoder，用于候选重排序。
- `models/medcpt/manifest.json`：固定模型 commit 和每个文件的 SHA256。
- `.venv-medcpt`：专用 Python 环境，复用本机 pytorch conda 环境的基础包。
- `requirements-medcpt.txt`：已验证的主要运行依赖版本。

首次下载（需要联网，不需要 API key）：

```powershell
.\.venv-medcpt\Scripts\python.exe scripts/medcpt_local.py download
```

无文章参数时可运行合成文本烟雾测试（不是医学或检索效果评测）：

```powershell
.\.venv-medcpt\Scripts\python.exe scripts/medcpt_local.py search
```

程序自动选择 PyTorch 可用的 CUDA，否则用 CPU。可指定 `--device cpu`。
模型逐个加载，降低同时占用的内存。不需要常驻服务或 FAISS。
按官方示例使用最后一层 CLS 向量与内积召回，不做余弦归一化；查询最长64 tokens，
文章编码最长512 tokens。论文段落按约400 tokens切块，重叠约50 tokens，保留原文位置。
空标题与正文块组成文献编码器的文本对，避免在每个文本块中重复注入含诊断答案的标题。
最多召回20个候选，再用交叉编码器排序，返回前 `--top-k` 个。
这里把原本的查询—文章模型迁移为查询—段落检索，尚需本地标注样本验证效果。

官方方法与权重：https://github.com/ncbi/MedCPT

## 在新机器重建

```powershell
python -m venv .venv-medcpt
.\.venv-medcpt\Scripts\python.exe -m pip install -r requirements-medcpt.txt
.\.venv-medcpt\Scripts\python.exe scripts/medcpt_local.py download
```

安装时的 PyTorch 构建需与目标 CPU/CUDA 平台兼容；当前机器仅验证了 CPU 推理。
模型和环境已由项目 `.gitignore` 排除，不应作为普通源码提交 GitHub。
