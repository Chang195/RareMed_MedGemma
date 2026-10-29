# 最终数据整理入口

使用主 README 中的 `pipeline.py`，避免运行过时的试跑命令。

```powershell
python pipeline.py diagnosis-candidates
python pipeline.py split
python pipeline.py export-sft
```

1. 从 collection_v3 的全部 figure 与 deepseek_collection_v3 的关联提案生成逐图状态。MRI 线索、单病例关联及作者报告确诊分别检查；没有提案、未决／鉴别诊断、动物或非患者图信号不升级为确诊样本。
2. 在所有可用图像上建立精确／相似重复关系；非候选图也能作为跨文献关系的桥接。按来源文章、已知曝光及保守连通组划分，不能只对 302 条诊断候选查重。
3. 使用 image + messages 导出；用户输入和溯源字段分开。已有 `diagnosis-sft-v0.2` 冻结为 239/24/33 条图像记录；新重建写入 diagnosis-sft-rebuild 并拒绝覆盖。

这些是作者报告诊断的弱监督研究数据，不是已完成独立临床标注、全局患者去重或隐私审核的金标准。原有粗 CADASIL 分组可能包含需进一步校正的名称；不要把原分组直接当成经过核验的二分类标签。三组混合实验使用另外的证据、确定性及像素文字筛查，审计保留在本地实验目录。

MedCPT 候选检索是可选文本整理分支，不是从原文生成新事实。来源证据与确诊背景存在，不表示图像本身足以作出诊断。

数据与字段格式见导出目录 manifest.json 和 provenance.jsonl；公开 GitHub 包不携带病例原文、图片或 API 响应。
