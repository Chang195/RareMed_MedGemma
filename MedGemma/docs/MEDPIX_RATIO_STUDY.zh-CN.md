# CADASIL 20% / 50% / 80% + MedPix：实验定义草案

用户已确认改用MedPix，MultiCaRe不进入本轮三组SFT。当前尚未定位本机MedPix包，数据来源版本待确认；没有启动新训练。

## 来源

原MedPix是NLM病例/影像数据库；论文作者发布的MedPix 2.0是其中CT/MRI病例的整理子集，不能混称为全部MedPix。已只读核查作者仓库CHILab1/MedPix-2.0，固定commit d3bc431576ec8041114806ec3eb58d75adf45029。

作者Zenodo record 12624810含Case_topic.json（3101414字节）、Descriptions.json（1269639字节）、images.zip（321936379字节）及splitted_dataset.zip。源记录声明CC BY 4.0，但仍需保留原始病例/图片归属与许可信息；本次核查不等于完成使用权审核。未下载完整图片。

## 三组设计（暂定，审核后冻结）

|组别|CADASIL图片|MedPix图片|合计|
|---|---:|---:|---:|
|20%|22|88|110|
|50%|55|55|110|
|80%|88|22|110|

这里比例指唯一训练图片/样本比例，不是患者比例、token比例或loss权重。三组固定总样本量和优化步数；不能每组都放88条CADASIL再按比例添加MedPix而不控制总量，否则数据量、训练量与比例同时变化。

该110样本方案还未被用户确认为最终规模。病例/文章组不可为凑数拆分；应在整组选择条件下检查能否满足嵌套22/55/88及88/55/22。若整组数量不允许精确比例，需报告实际比例，或事先另行定义按曝光次数的采样实验，不能静默用重复样本假充新增独立样本。

## 任务与审核

- 优先采用相同的“图片→作者报告诊断”任务、相同问题及同粒度标签。MedPix中的Case Diagnosis/Case Title仅是候选来源字段，必须检查其与该图片的病例关联，不能自动把疾病讨论或鉴别诊断当作确诊目标。
- 用户输入仅图片及固定问题；History/Findings/Case Diagnosis/Topic等都不得拼入图片诊断输入。诊断证据保留在provenance。
- 检查MedPix是否含CADASIL/NOTCH3或重复现有病例，单独暂存；不能默认MedPix就是非CADASIL金标准阴性。
- 优先筛选可比的脑MRI病例。若混入大量其他部位CT/描述任务，必须明确这会混入来源、模态和任务差异，不能只解释为疾病比例效应。
- 图像解码、文件/像素哈希、病例UID、病例多图、图注/诊断关联、原始标注、PHI/OCR和许可均需记录与审核。
- 如果来源包提供现成划分，先检查其定义和病例重叠，不直接将所有图合并。固定共同的MedPix验证/测试来源后，三组仅从训练侧抽样。

## 训练与评价

三组都从保存的原始google/medgemma-1.5-4b-it revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b重新创建独立LoRA；不用既有SFT adapter热启动。沿用NF4/FP32/SDPA及assistant-only loss，其他预算在训练前统一冻结。

固定相同验证集选择checkpoint，固定相同测试集、推理和解析规则比较。现有test已经看过，本轮可用作探索性迭代对比，确认性结论需要新的未查看测试病例/来源。公开QA仅作指定能力的回归检查，需预先规定可接受下降范围，不用p>0.05声称没有退步。

参考：
- https://catalog.data.gov/dataset/medpix
- https://github.com/CHILab1/MedPix-2.0
- https://zenodo.org/records/12624810
