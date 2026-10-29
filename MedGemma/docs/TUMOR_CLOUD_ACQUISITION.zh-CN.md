# GBM / 星形细胞瘤：官方 PMC 云存储图片获取

适用范围：已冻结的单页前 1000 条检索，经历史排除和原规则全文筛选的两组候选。
不重新检索、不扩展疾病范围、不调用付费 API。普通 PMC 文章 HTML 获取被禁用。

## 获取层变化

PMC 官方 2026 年分发方式已迁移到公开 S3 云存储：
https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
https://pmc-oa-opendata.s3.amazonaws.com/README.txt

按 PMCID 列出文章版本，获取 JSON 元数据、JATS XML 和对应媒体文件。
不用 AWS 账号、Access Key 或付费 API。
保留原文 BioC 图注、偏移、全文、病例抽取 prompt、schema、MRI 图注规则和许可过滤。
只用 JATS 的同图号/原文件名定位官方媒体，不猜扩展名、不拆分面板、不裁剪/去箭头。
媒体 URL 的官方 MD5（存在时）及本地 SHA-256 都校验。

版本优先匹配 BioC 显式版本。如果 BioC 没有版本，只允许唯一云端版本，且 DOI、图号、
图注文本一致。多版本、缺失文件或图注映射不清全部 held，不猜测。
图注匹配仅为核验做 Unicode/空白规范化，不改写原文或标准答案。
官方原图与此前网页呈现文件可能不是相同编码；记录下载来源、哈希和版本，不声称字节等同。
图片获取成功不等于临床审核或再分发授权完成。

## VS Code

在 RareMed_MedGemma 终端，先后运行（不要同时运行）：

```powershell
.\.venv\Scripts\python.exe run_tumor_images.py gbm collect
.\.venv\Scripts\python.exe run_tumor_images.py astrocytoma collect
```

每条完成后再执行下一条。默认沿用原 max_articles=1000 和所有 fulltext_passed 队列，
当前为 GBM 571 篇、星形细胞瘤 590 篇。此阶段均为文献/图像候选，不是确诊病例。

共享 pmc-cloud-cache-v1 会复用同 URL 的元数据、XML 和媒体；读取前核验哈希。
原 acquire_screened.py 执行断点续跑并保留状态。访问错误暂存，不当作阴性。
如出现代码/清单哈希不匹配，停止并检查，不删除 checkpoint 或关掉校验。

完成后看各疾病目录 reports/collection_v3_receipt.json 与
data/processed/collection_v3_figures.jsonl。
普通图像记录仍使用原 schema，新增 acquisition_channel 与 cloud_mapping_receipt。
raw/<PMCID>/cloud_mapping_receipt.json 保存元数据、JATS、版本和图号映射证据。

待确认批量采集结果后，再运行无付费的模型输入准备：

```powershell
.\.venv\Scripts\python.exe run_tumor_images.py gbm prepare-inputs
.\.venv\Scripts\python.exe run_tumor_images.py astrocytoma prepare-inputs
```

这两个命令也不调用 DeepSeek。后续抽取需要独立预检并显式允许付费调用。

## 已验证边界

云端小样本验证与批量结果分开保存 collection_cloud_smoke_v3；
smoke 在原图注 MRI 规则命中的文章中选前三篇，仅验证获取流程。
正式 collect 不按 smoke 成功与否选择病例。
4 项离线测试覆盖官方 URL 限制、精确文件映射、MD5 失败拒收和缓存复用。
