# 在VS Code运行三组MedPix比例SFT

打开RareMed_MedGemma项目根目录（能看到mixture_sft.py），新建终端。所有命令为Python，不需要.ps1或激活环境。

## 数据和结果位置

默认直接读取你放在项目中的：

```text
$PROJECT_ROOT\RareMed_CADASIL_SFT_Mixture_Experiments
```

原始模型继续使用已保护的G盘固定版本，结果默认保存在：

```text
G:\RareMed_MedGemma_Local\runs\mixture-ratios-v1
```

VS Code是运行代码的工具；数据可以在项目内，权重和结果可以在其他本地磁盘。该入口不移动或改写你的原始数据。

## 执行顺序

先准备（不训练）：

```text
python mixture_sft.py prepare
```

若本轮已经显示PREPARE PASS，无需重复。它核对原始权重、三组分组/哈希，并检查全部train/val的实际processor掩码。原始JSONL不改；单独的loader-data视图只把图片路径解析为绝对路径、把val文件名映射为validation，目标和消息不变。

开始三次独立SFT（按A→B→C顺序）：

```text
python mixture_sft.py train --arm all
```

每组先从原始模型新建一次性LoRA，对正负各1条train样本做有限loss、反向和梯度检查，不执行优化步。通过后丢弃探针adapter，再从原始模型新建正式LoRA训练。因此B不接着A训练、C不接着B训练，也不会使用上轮diagnosis-night-01或smoke adapter。

每组100条，3轮，batch=1、梯度累积4，即25步/轮、75优化步/组，共225优化步。NF4 double quant、FP32、SDPA，r=4、alpha=8、dropout=0，学习率2e-5，AdamW及梯度裁剪沿用旧代码。仅assistant正文计算loss，保留既有结束token屏蔽规则。没有改用全参数训练或修改原图。

初始及每轮末只计算共同validation loss，以最低平均样本assistant-token NLL选择best.json。该验证集5阳性/62其他来源标签，因此这个选择标准受验证集分布影响，不是敏感度/特异度，也不保证最佳临床区分能力。三组固定同一选择标准，不能根据test另挑checkpoint。

若希望手动分开运行：

```text
python mixture_sft.py train --arm A
python mixture_sft.py train --arm B
python mixture_sft.py train --arm C
```

一次执行一条并等待结束，不同时运行。

## 状态与恢复

另一个终端查看：

```text
python mixture_sft.py status
```

若中断，使用：

```text
python mixture_sft.py resume --arm all
```

它跳过已COMPLETE组，恢复有完整checkpoint的组，尚未开始的组从原始模型开始。每组在第1步、每20步和每轮末保存；恢复不覆盖完整checkpoint。不要在prepare后修改源代码、数据或环境，指纹检查会拒绝变化。

## 可验证产物

每组独立位于结果根目录A/、B/、C/：config.json、token-checks.json、probe.json、training.jsonl、validation-history.json、best.json、latest.json、checkpoint-*/、status.json，以及源码副本。checkpoint含LoRA与优化器/RNG/样本游标，原始模型不合并。

## 已知限制

- 32张JPEG的上游RGB像素指纹未在本机复现，但所有原始文件SHA256一致，本机像素重复检查通过；该差异完整记录在config/data-check中，未改原图或降级训练环境。
- 三组、原始模型的后续比较必须统一使用同一本机Pillow/processor；这保证本机比较条件一致，不等于跨机器像素完全复现。
- 阳性来自RareMed，训练负例来自MedPix，存在来源与疾病标签混杂；数据仍非独立临床审核金标准。
- 共同test68条、CADASIL6张且已包含此前看过的病例；这是探索性实验。正式测试输出不在训练命令中生成。
- 必须用新二分类问题建立新的原始模型基线，不能复用旧开放诊断指标。

当前三组训练入口是对已验证训练核心的适配，尚未完成整组三轮实跑。先观察train-only probe PASS与Step 1/75及checkpoint保存，再让其持续运行。若失败会停止，保留日志，不自动换模型或修改配置。
