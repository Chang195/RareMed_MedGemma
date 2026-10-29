"""Fixed, conservative output parsing and paired statistics; no model calls."""
import math
import random
import re
from collections import Counter

VERSION='diagnosis-strict-v1'
FULL_NAME=r'cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy'

def normalize(text):
    text=re.sub(r'<[^>]+>',' ',text).replace('**','').replace('`','')
    return re.sub(r'\s+',' ',text).strip().strip('"\'').rstrip('.').strip().lower()

def parse_diagnosis(raw,other_labels):
    text=normalize(raw)
    text=re.sub(FULL_NAME+r'\s*(?:\(cadasil\))?','cadasil',text,flags=re.I)
    text=re.sub(r'cadasil\s*\(cadasil\)','cadasil',text)
    if not text: return 'unparsed'
    if re.search(r'\b(possible|possibly|probable|probably|suspected|unlikely|could|might|may|uncertain|unsure|differential|consider|either|versus|vs)\b|cannot|can.t|unable|insufficient|not enough|don.t know|rule out|\bor\b|/',text):
        return 'uncertain_multi'
    answer=re.sub(r'^(?:the )?(?:most likely )?(?:diagnosis|answer)\s*(?:is\s+|:\s*)','',text).strip()
    if answer=='cadasil': return 'positive'
    if re.fullmatch(r'(?:not cadasil|no cadasil|no evidence of cadasil|cadasil (?:is )?(?:excluded|ruled out))',answer):
        return 'negative'
    if re.search(r'\bcadasil\b',text): return 'mention_only'
    if answer in {normalize(x) for x in other_labels}: return 'other'
    return 'unparsed'

def ratio(n,d): return dict(numerator=n,denominator=d,value=n/d if d else None)

def binary_metrics(tp,fn,fp,tn):
    sensitivity=ratio(tp,tp+fn); specificity=ratio(tn,tn+fp)
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,sensitivity=sensitivity,specificity=specificity,
        precision=ratio(tp,tp+fp),F1=ratio(2*tp,2*tp+fp+fn),
        balanced_accuracy=(sensitivity['value']+specificity['value'])/2 if sensitivity['value'] is not None and specificity['value'] is not None else None)

def diagnosis_metrics(rows):
    tp=fn=fp=tn=up=un=0
    for r in rows:
        positive=r['reference_cadasil']; category=r['category']
        if category not in ['positive','negative','other']:
            if positive: up+=1
            else: un+=1
        elif positive:
            if category=='positive': tp+=1
            else: fn+=1
        else:
            if category=='positive': fp+=1
            else: tn+=1
    p=tp+fn+up; n=tn+fp+un
    # Abstentions stay separate; do not silently count them as true negatives.
    return dict(total=len(rows),reference_CADASIL=p,reference_other_source_labels=n,
        categories=dict(Counter(r['category'] for r in rows)),unresolved_reference_CADASIL=up,
        unresolved_reference_other=un,coverage=ratio(tp+fn+fp+tn,len(rows)),
        resolved_only=binary_metrics(tp,fn,fp,tn),
        all_sample_sensitivity_bounds=[tp/p,(tp+up)/p] if p else None,
        all_sample_specificity_bounds=[tn/n,(tn+un)/n] if n else None,
        interpretation='TP/FN/FP/TN and point metrics apply to resolved outputs only; unresolved retained with full-reference bounds. Other labels are not adjudicated clinical negatives.')

def paired_qa(base,sft,repeats=10000):
    assert [r['sample_id'] for r in base]==[r['sample_id'] for r in sft]
    assert all(a['reference']==b['reference'] for a,b in zip(base,sft))
    deltas=[int(b['correct'])-int(a['correct']) for a,b in zip(base,sft)]
    n=len(deltas); improved=deltas.count(1); regressed=deltas.count(-1)
    # Difference bootstrap depends only on the counts of -1/0/+1; numpy is an existing dependency.
    import numpy as np
    draws=np.random.default_rng(42).multinomial(n,[regressed/n,1-(regressed+improved)/n,improved/n],size=repeats)
    samples=(draws[:,2]-draws[:,0])/n
    lo,hi=np.quantile(samples,[0.025,0.975]).tolist()
    discordant=improved+regressed
    if discordant:
        k=min(improved,regressed)
        # Binomial exact McNemar p-value via log-space terms (no scipy dependency).
        logs=[math.lgamma(discordant+1)-math.lgamma(i+1)-math.lgamma(discordant-i+1)-discordant*math.log(2) for i in range(k+1)]
        peak=max(logs); pvalue=min(1.0,2*math.exp(peak)*sum(math.exp(x-peak) for x in logs))
    else: pvalue=1.0
    return dict(n=n,base_accuracy=ratio(sum(r['correct'] for r in base),n),
        sft_accuracy=ratio(sum(r['correct'] for r in sft),n),delta_accuracy=sum(deltas)/n,
        improved=improved,regressed=regressed,unchanged=n-improved-regressed,
        paired_question_bootstrap_95CI=[lo,hi],bootstrap_seed=42,bootstrap_repeats=repeats,
        mcnemar_exact_two_sided_p=pvalue,
        conclusion='Descriptive paired comparison only. No non-inferiority margin was specified; nonsignificance is not evidence of no degradation.')
