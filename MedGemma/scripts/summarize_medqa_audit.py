import os
from pathlib import Path
from pathlib import Path
import json,hashlib,csv,re
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'medqa-audit-v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
p=read(ROOT/'mixture-evaluation-v1/protocol.json');proxy=read(OUT/'original_score_audit.json');small=read(OUT/'summary.json')
assert (OUT/'full-answer-summary.json').exists(),'Finish full-answer diagnostic first.'
full=read(OUT/'full-answer-summary.json');fast=read(OUT/'fast-tokenizer-check.json')
format_review=[]
for j,row in enumerate(full['results']):
    raw=read(OUT/f'full-answer-{j:02d}.json')['raw_output']
    tail=raw.rsplit('Final Answer:',1)[-1] if 'Final Answer:' in raw else ''
    boxed=re.findall(r'\\boxed\s*\{\s*([ABCD])\s*\}',tail)
    reviewed=row['final_prediction'] or (boxed[-1] if boxed and not row['truncated'] else '')
    format_review.append(dict(**row,terminal_format_review=reviewed))
full['terminal_format_review']=dict(disclosure='Additional post-hoc serialization review after observing boxed A in the first sample; exact boxed A-D in the final-answer suffix only. Not a prespecified benchmark score.',results=format_review)
raw=Path(str(Path(os.environ.get('RAREMED_LOCAL_ROOT', str(Path.home() / 'RareMed_MedGemma_Local'))) / 'huggingface/hub/datasets--GBaker--MedQA-USMLE-4-options/snapshots/0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18/phrases_no_exclude_test.jsonl'))
a=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines() if x.strip()]
b=[json.loads(x) for x in (ROOT/'mixture-evaluation-v1/datasets/medqa.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
assert len(a)==len(b)==1273
assert all(x['question']==y['question'] and [x['options'][c] for c in 'ABCD']==y['choices'] and x['answer_idx']==y['reference'] and x['answer']==x['options'][x['answer_idx']] for x,y in zip(a,b))
direct=[read(f) for f in sorted(OUT.glob('sample-*.json'))]
assert all(r['first_step']['top10'][0]['token_id']==r['same_prompt_generation']['first_generation_token_id'] for r in direct)
first_tokens=[r['first_step']['top10'][0]['decoded'] for r in direct]
result=dict(status='COMPLETE local failure-mechanism audit; official accuracy not reproduced',
 original_model=p['model_id'],revision=p['revision'],quantization=p['quantization'],
 confirmed_fault='Original scorer ranks bare A-D at the first assistant position even when the model is beginning analysis/format tokens rather than giving an answer.',
 first_token_argmax_matches_actual_generation=True,fixed_train_sample_first_tokens=first_tokens,
 original_four_model_proxy_results=proxy,train_split_diagnostic=small,extended_same_prompt_diagnostic=full,
 data_normalization_all1273_verified=True,fast_slow_input_ids=fast,
 excluded=['Wrong normalized gold letter/option order on1273 questions','Different saved processor vocabulary/template','Failure to use the original Base checkpoint instead of an adapter'],
 not_excluded=['NF4 quantization contribution versus unquantized inference','Other official prompt/decoding/answer-extraction differences','Generalization accuracy under a properly validated complete-answer protocol'],
 consequence='Existing30.6%-30.8% values are first-token proxy accuracies, not validated completed-answer MedQA accuracy. They do not establish knowledge retention; rerun all checkpoints under a separately frozen validated protocol.',
 distinction='The end_of_turn correction affects the separate diagnostic parser; original MedQA proxy has no freely generated answer to repair by parsing.',
 suggested_next_steps=['Validate complete-answer generation and parser on development/training data, recording truncation and unparsed cases.',
   'Set a fixed adequate generation budget before official test inference; do not choose prompt/checkpoints by test accuracy.',
   'If resources allow, compare NF4 and unquantized inference on identical development items.',
   'Rerun Base/A/B/C with identical newly frozen protocol, retaining all original proxy results and disclosing the post-hoc audit.'])
(OUT/'audit_report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
with (OUT/'complete_answer_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
 rows=full['results'];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
examples='\n'.join(f"| {r['source_index']} | {r['original_bare_prediction']} | {r['final_prediction'] or '未解析'} | {r['terminal_format_review'] or '未解析'} | {r['gold']} | {r['output_tokens']} | {r['truncated']} |" for r in format_review)
(OUT/'README.zh-CN.md').write_text(f'''# MedQA 异常排查报告

## 结论

已复现原快速评测的设计问题：它在 assistant 回答的第一个位置比较不带空格的 A/B/C/D 概率；模型在该位置实际开始生成分析文字或 thought 标记。这与从完整回答提取最终选项不是同一个测量。

Base 在原 1,273 题中选择 A 947 次（74.39%）。不带空格四字母的概率总和均值为 7.18%，996/1,273 题低于10%。因此原方法经常在低概率的四个 token 中强行选择。原 raw_output 字段存的是评分器选出的字母，不是模型自由生成的完整回答；不能仅重解析旧文件恢复真实完整答案。

## 实际对照

- 下载固定版本的公开 MedQA 训练划分，seed42 预先固定8题，排除与本地测试集题目文本完全相同的记录。未用测试集调提示词。
- 同一原始提示词、同一 NF4 原始 Base：8/8 的首 token 全词表 argmax 与实际生成首 token 相同，但都不是 A/B/C/D；原裸字母评分8/8都选A。这排除了“生成与前向取得不同首 token”这一解释，暴露评分位置的语义不匹配。
- 仅将候选 token 换成带空格的字母并没有得到有效答案评分：8题中的四字母概率质量平均约 0.00000128，不能把这种替换直接当成修复。
- 64-token 原提示词生成8/8截断；256-token 论文风格生成3/4截断。这些未完成输出不能当作最终答案，更不能据此报告0%能力。
- 对最早已固定的前2题，只增加生成预算到1536，不改变原始user提示词，得到以下独立记录。

| 训练集行号 | 原首位置评分 | 预设自动解析 | 最终格式复核 | 来源答案 | 生成tokens | 截断 |
|---:|---|---|---|---|---:|---|
{examples}

第一题最终使用LaTeX boxed格式，预设解析器未覆盖。观察此格式后另做终止答案后缀中的精确boxed字母复核，标为事后技术格式审核，不更改已冻结的自动解析结果，也不将其计为正式benchmark。正式重评前必须提前固定对此格式的处理。

这2题和8题均为训练划分上的机制验证，不是独立 MedQA 成绩，不能推算官方69.1%的复现率。完整回答的原始文本、token ID、输入 ID 和耗时均另存，不覆盖旧预测。

## 已核对和仍未知的内容

全部1,273题的题目、选项顺序、答案字母和答案文本与下载源一致。原始snapshot与保存的处理器词表和模板一致。另做了8题fast/slow tokenizer输入ID比较，结果为：{fast['all_match']}。原始Base不加载adapter。

尚未做非量化对照，不能排除NF4贡献，也不能声称全部38.5个百分点差距均由单一原因解释。本次论文风格提示词是方法学对照，不是逐字、完整官方评测复现。

诊断任务的终止token解析修正与本次MedQA首位置评分问题不同。此排查不否定已核对的诊断混淆矩阵，但会限制PPT里关于医学QA知识保持的解释。原MedQA数值只能称为首token快速评分协议的结果，不能用来证明其他能力未退化。

## 后续

先在开发/训练数据上验证完整生成与最终答案提取，固定足够预算、截断/未解析处理规则和模型版本，再统一重评Base/A/B/C。保留旧协议与旧成绩；新协议不得根据测试集成绩反复调参。完整生成成本明显高于一次前向，需按实测tokens估算时间。

官方参考：[模型卡](https://huggingface.co/google/medgemma-1.5-4b-it)、[MedGemma技术报告的文字评测模板](https://arxiv.org/html/2507.05201v3#A4)、[1.5技术报告](https://arxiv.org/html/2604.05081)。
''',encoding='utf-8')
print(json.dumps(dict(full_answer_results=full,fast_tokenizer_match=fast['all_match']),indent=2))
