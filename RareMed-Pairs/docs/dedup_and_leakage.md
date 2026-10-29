# 去重与数据泄漏检查

不调用API、不删除原图，不进行PHI脱敏。

```powershell
python scripts/dedup_leakage.py build
Invoke-Item ".\data\datasets\dedup_v1_1\review.html"
```

读取formal_candidates_v1的全量867条索引，当前输出到 `data/datasets/dedup_v1_1/`；保留早期dedup_v1结果。

## 去重层次

- 文件SHA256相同：同一文件。
- EXIF方向规范化后的RGB尺寸和像素哈希相同：同一解码图像，可发现仅容器/无损编码不同的图片。
- 63bit DCT感知哈希：比较原图、水平镜像、中心保留90%/75%的图像；配合dHash召回相似候选。
- 文字：长度至少12词的图注，以规范化文本与词三元组Jaccard寻找高度相似候选。

感知哈希对相似脑解剖、黑色背景、模板图存在误报，也可能漏掉任意面板裁剪、
大幅旋转、遮挡等变化。图注相同也可能是通用表述。候选不等于同图，更不等于同患者。
阈值是未经标注集校准的启发式，报告中不宣称去重准确率。

仅精确文件/像素关系或后续明确duplicate决定用于同图分组；保留所有来源记录，
通过canonical_image_sample_id给出索引代表，不删除不同图注/标签。

## 防泄漏分组

同一来源文章、精确同图、确认的患者ID、已记录的衍生关系放同组。
尚未解决的图像/图注相似关系也保守地放同组，以降低已检测关系跨集合风险，
但可能将不同患者过度合并。不能把最终组数当成患者数。

`configs/known_model_exposure.json`记录已用于QLoRA smoke的两张图片。
它们所在整个连通组强制建议进入train，不能当作从未接触的验证/测试样本。
后续所有调参、训练或人工挑选所接触的数据需要持续更新此清单。

proposed_split按分组与固定seed约80/10/10生成，未做临床分层；
split仍为unassigned，所有training_ready为false。候选队列之外为hold。
这不是冻结的最终训练集，也不是独立金标准测试集。

## 输出

- `records.jsonl`：全量记录＋同图索引＋保守分组＋建议划分。
- `duplicate_pairs.jsonl`：候选对、触发原因、相似度和是否参与保守分组。
- `groups.json`：连通组。
- `review.html`：前60对的图片/图注对照，完整结果在JSONL中。
- `draft_split_audit.json`：建议分组的检测结果。
- `summary.json`：哈希、数量及局限。

若以后审核部分候选，可另建决策JSON：

```json
{
  "source_manifest_sha256": "inventory.jsonl的SHA256",
  "decisions": [
    {"sample_a":"样本ID1","sample_b":"样本ID2","decision":"uncertain","reason":"待确定"}
  ]
}
```

decision可为duplicate、same_case、unrelated、uncertain；unrelated不撤销精确同图关系。
使用 `--decisions 文件路径 --output 新目录` 重建，保留旧版本。

## 集合与答案泄漏检查

建议划分检查：

```powershell
python scripts/dedup_leakage.py audit --dataset data/datasets/dedup_v1_1/records.jsonl --relations data/datasets/dedup_v1_1/duplicate_pairs.jsonl --split-field proposed_split
```

检查实际SFT导出：

```powershell
python scripts/dedup_leakage.py audit --dataset data/datasets/caption-pilot-v0.1/sft_pilot.jsonl --report reports/caption_pilot_leakage.json
```

检查来源、分组、图像哈希、重新计算的解码像素哈希、核验患者ID、长目标文本和衍生关系跨集合；
核对已知训练曝光样本、文章及其连通组是否进入验证/测试，曝光清单也可通过--exposure指定。
检查目标答案是否原样出现在用户输入，并对部分诊断/基因线索给出警告。
检查只读取messages内用户/assistant内容，不把metadata中的溯源证据误当成模型输入。
故障退出码为1；只有unassigned或单集合时明确标为split_check_not_applicable，不报划分通过。

正式SFT导出必须继承split_group_id、image_sha256、pixel_sha256（如适用）、父样本ID，
并提供duplicate_pairs关系文件。只有原图路径和messages的独立JSONL，不能检验未知的感知相似关系；
未提供关系文件时明确警告。未识别的跨论文同患者、预训练数据污染、同义改写泄漏等仍可能存在。
图像级去重与分组不解决病例关联、临床标签、脱敏或发布许可问题。
