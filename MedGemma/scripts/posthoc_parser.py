"""Post-hoc top-1 diagnosis extraction. Never uses the reference or model identity."""
import re
from eval_scoring import FULL_NAME,normalize

VERSION='top1-extraction-v2-posthoc'
HEDGE=r'\b(possible|possibly|probable|probably|suspected|unlikely|could|might|may|uncertain|unsure|either|versus|vs)\b|cannot|can.t|unable|insufficient|don.t know|rule out|\bor\b|/'
NON_DIAGNOSES={'mra','magnetic resonance angiography','mra (magnetic resonance angiography)',
 'mri','magnetic resonance imaging','ct','computed tomography','brain mri','normal','unknown'}

def canonical(text):
    text=normalize(text)
    text=re.sub(FULL_NAME+r'\s*(?:\(cadasil\))?','cadasil',text,flags=re.I)
    return re.sub(r'cadasil\s*\(cadasil\)','cadasil',text)

def extract(raw,truncated=False,other_labels=()):
    result=dict(category='unparsed',candidate=None,evidence=None,evidence_span=None,
        extraction_basis=None,flags=[],review_required=True)
    # Prefer an explicit answer assertion, not arbitrary bold findings in an explanation.
    assertion=re.search(r'\b(?:(?:the|a)\s+)?(?:most\s+likely\s+)?diagnosis\s*(?:is\s+|:\s*)',raw,re.I)
    if assertion:
        start=assertion.end(); tail=raw[start:]
        bold=re.match(r'\s*\*\*(.+?)\*\*',tail,re.S)
        if bold:
            candidate=bold.group(1).strip(); begin=start+bold.start(1); end=start+bold.end(1)
        else:
            stop=re.search(r'[.!?\n]|<end_of_turn>|<eos>',tail)
            if stop is None and truncated:
                result['flags'].append('truncated_candidate_not_delimited'); return result
            segment=tail[:stop.start()] if stop else tail
            candidate=segment.strip(); begin=start+len(segment)-len(segment.lstrip()); end=begin+len(candidate)
        result.update(candidate=candidate,evidence=raw[assertion.start():end],
            evidence_span=[assertion.start(),end],extraction_basis='first_explicit_diagnosis_assertion')
    else:
        # Bare labels must match the pre-frozen vocabulary, or an explicit CADASIL denial.
        first=re.search(r'[^\r\n]+',raw)
        if not first: return result
        candidate=first.group().strip(); name=canonical(candidate)
        known={canonical(x) for x in other_labels}|{'cadasil'}
        denial=re.fullmatch(r'(?:not cadasil|no cadasil|no evidence of cadasil|cadasil (?:is )?(?:excluded|ruled out))',name)
        if name not in known and not denial:
            if re.search(HEDGE,name,re.I): result['category']='uncertain_multi'
            elif re.search(r'\bcadasil\b',name): result['category']='mention_only'
            return result
        result.update(candidate=candidate,evidence=first.group(),evidence_span=list(first.span()),
            extraction_basis='standalone_known_label_or_explicit_denial')
    name=canonical(result['candidate'])
    if not name: return result
    if re.search(HEDGE,name,re.I): result['category']='uncertain_multi'; return result
    if name in NON_DIAGNOSES:
        result['flags'].append('candidate_is_modality_or_nondiagnostic_answer'); return result
    if name=='cadasil': category='positive'
    elif re.fullmatch(r'(?:not cadasil|no cadasil|no evidence of cadasil|cadasil (?:is )?(?:excluded|ruled out))',name): category='negative'
    elif re.search(r'\bcadasil\b',name): category='mention_only'
    else: category='other'
    result['category']=category
    # Repetition of one bare diagnosis is a format failure, but top-1 remains identifiable.
    if result['extraction_basis']=='standalone_known_label_or_explicit_denial':
        lines=[canonical(x) for x in raw.splitlines() if x.strip() and not x.strip().startswith('<')]
        if len(lines)>1:
            if all(x==name for x in lines): result['flags'].append('repeated_same_label')
            else:
                result['flags'].append('extra_text_after_bare_label_requires_review')
                if re.search(r'\b(?:not cadasil|no evidence of cadasil|cadasil is ruled out)\b',canonical(raw)):
                    result['category']='uncertain_multi'
    remaining=raw[result['evidence_span'][1]:]
    if truncated: result['flags'].append('explanation_truncated_candidate_complete')
    if re.search(r'however|without further|cannot|differential|other possibilities|definitive diagnosis',remaining,re.I):
        result['flags'].append('later_caveat_or_alternatives_requires_review')
    if category=='positive' and re.search(r'(?:not|no evidence of)\s+cadasil|cadasil\s+(?:is\s+)?(?:excluded|ruled out)',remaining,re.I):
        result['category']='uncertain_multi'; result['flags'].append('later_cadasil_negation')
    later=list(re.finditer(r'\bdiagnosis\s*(?:is\s+|:\s*)',remaining,re.I))
    if later:
        result['category']='uncertain_multi'; result['flags'].append('multiple_diagnostic_assertions_require_review')
    # Every automatic v2 decision remains provisional until independent text extraction review.
    return result
