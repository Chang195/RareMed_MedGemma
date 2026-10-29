# Orphadata：先建立可溯源的疾病知识子图

入口：`scripts/orphadata_kg.py`。只依赖 Python 标准库，不需要 GPU、API 密钥或大模型。

在 VS Code 打开 RareMed-Pairs 文件夹，在该文件夹的终端运行：

```powershell
python scripts/orphadata_kg.py build --query CADASIL
```

首次下载三个官方英文 XML，合计约 74 MiB。后续校验 SHA-256 并复用，不重复下载。

已有缓存时可完全离线重建：

```powershell
python scripts/orphadata_kg.py build --query CADASIL --offline
```

打开结果：

```powershell
Invoke-Item ".\data\knowledge\orphadata\ORPHA136_v1\report.html"
```

## 输出是什么

- `data/knowledge/orphadata/raw/`：官方源文件与下载回执，保留 XML 元数据、许可声明、URL、时间和哈希。
- `data/knowledge/orphadata/ORPHA136_v1/disease.json`：疾病、全部表型及原始频率、基因、发病年龄和遗传方式。
- `graph.json`、`nodes.jsonl`、`edges.jsonl`：可供后续检索的节点和关系，每条关系保存来源文件哈希与原始记录证据。
- `phenotypes.csv`：可在 Excel 中打开的表型清单。
- `summary.json`、`manifest.json`：统计、输入和输出哈希。
- `report.html`：先给人查看这一步是否成功的报告。

这是疾病层面的知识，不是新病历、患者标签或 MRI 检查结果。未调用大模型，未运行合成与训练。HPO 术语只使用源文件提供的关联，不自动推断父子层级或完整 HPO 本体。

## 查询其他疾病与固定版本

```powershell
python scripts/orphadata_kg.py search --query CADASIL --offline
python scripts/orphadata_kg.py build --orpha-code 136 --offline
```

名称匹配有多个结果时停止，不猜测病种。可以改用来源 ORPHA 编号。这里的查询只覆盖表型源文件列出的疾病，未匹配不代表 Orphanet 整体没有该疾病。

如需更新源版本，使用新的缓存和输出目录，旧版不覆盖：

```powershell
python scripts/orphadata_kg.py build --query CADASIL --cache-dir data/knowledge/orphadata/raw_v2 --output data/knowledge/orphadata/ORPHA136_v2
```

## 知识使用边界

- 原文频率例如 `Frequent (79-30%)` 按原样保存，不默认取中点、不视为精确概率。
- 未记录的表型、频率、自然史信息均保持缺失，不当作阴性。来源明确标为排除的表型使用 `excludes_phenotype`。
- 基因关系类型与 `Assessed` 等状态原样保留，不能把所有关联都升级为致病证明。
- 文件导出日期和记录核验日期分别保存。新文件不等于所有临床关系都刚被更新。
- 不根据图谱自动推断精确年龄、外显率、症状共现概率、MRI 序列或部位规则。
- 后续合成需区分疾病知识、真实病例事实和虚拟患者事实。此步骤不改变现有训练／验证／测试划分，也不把测试病例交给生成器。

## 官方来源

- https://sciences.orphadata.com/orphanet-scientific-knowledge-files/
- https://sciences.orphadata.com/phenotypes/
- https://sciences.orphadata.com/genes/
- https://sciences.orphadata.com/natural-history/

Orphadata 页面声明 CC BY 4.0；原文件许可也保存在回执和输出中。发布派生数据时保留 Orphanet / INSERM 署名、来源、版本和修改说明。

## 检查

```powershell
python -m unittest discover -s scripts -p test_orphadata_kg.py
```

检查未知频率、显式排除、候选基因关系、歧义病名、损坏 XML 与缓存完整性等边界。
