"""Conservative source-span routing; these rules do not establish clinical truth."""
import re

PATTERNS = {
    'presentation': r'\b(?:present(?:ed|s|ing)?\s+(?:with|to)|complain(?:ed|s)?\s+of|symptoms?\s+(?:began|started|developed)|developed|noticed|onset|dysarthria|headache|migraine|weakness|seizures?|cognitive\s+decline)\b',
    'medical_history': r'\b(?:past\s+medical\s+history|medical\s+history|history\s+of|previous(?:ly)?|prior\s+(?:stroke|episode)|hypertension|diabetes|smok(?:ed|ing)|tobacco|alcohol)\b',
    'family_history': r'\b(?:family\s+history|familial|mother|father|sister|brother|parents?|grandmother|grandfather)\b',
    'demographics': r'\b(?:\d{1,3}[\s\-–‑]+years?[\s\-–‑]+old|aged\s+\d{1,3})\b',
    'imaging_description': r'\b(?:MRI|MR|magnetic\s+resonance|FLAIR|T[12](?:WI|W)?|DWI|ADC|SWI|hyperintens\w*|hypointens\w*|signal\s+intensity|white\s+matter|infarct\w*|microbleeds?|lacunes?)\b',
    'diagnostic_results': r'\b(?:identified|detected|revealed|confirmed|demonstrated|showed|found|positive|negative|consistent\s+with|diagnos\w*|pathogenic|heterozygous|homozygous|variant\s+of\s+uncertain)\b',
    'genetic_results': r'\b(?:gene(?:tic)?|NOTCH3|variant|mutation|sequencing|biopsy|histolog\w*|osmiophilic)\b',
}
RX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
TREATMENT = re.compile(r'\b(?:administered|prescribed|commenced|started\s+on|treated\s+with|aspirin|clopidogrel|atorvastatin|lansoprazole|mg\b|therapy)\b', re.I)
CT = re.compile(r'\b(?:CT|computed\s+tomograph\w*|computerised\s+axial\s+tomograph\w*)\b', re.I)
MRI = re.compile(r'\b(?:MRI|MR|magnetic\s+resonance|FLAIR|T[12](?:WI|W)?|DWI|ADC|SWI)\b', re.I)
DIAGNOSTIC_LEAK = re.compile(r'\b(?:CADASIL|NOTCH3|pathogenic|heterozygous|homozygous|genetic\s+(?:test\w*|counsell\w*)|diagnos\w*)\b', re.I)
GENERAL = re.compile(r'\b(?:patients\s+with|patients\s+(?:were|are)|controls?|cohort|participants?|subjects?|mean\s+age|median\s+age|\d+\s+patients|should\s+be|is\s+recommended|can\s+(?:cause|present))\b', re.I)


def sentence_spans(chunk):
    text = chunk['text']
    protected = re.sub(r'\b(?:Fig|Figs|Dr|Mr|Mrs|Ms|Prof|vs)\.', lambda m: m[0][:-1] + '\u2024', text)
    begin = 0
    cuts = [m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])', protected)] + [len(text)]
    for end in cuts:
        a, b = begin, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b-1].isspace():
            b -= 1
        if a < b:
            base = chunk.get('start_in_passage', 0)
            yield {**chunk, 'text': text[a:b], 'start_in_passage': base+a,
                   'end_in_passage': base+b, 'status': 'source_span_unreviewed'}
        begin = end


def route(text, field):
    """Return routing decision and reasons; no numeric relevance threshold is implied."""
    clinical = {'presentation', 'medical_history', 'family_history', 'demographics'}
    if field in clinical:
        if DIAGNOSTIC_LEAK.search(text):
            return False, ['diagnosis_or_genetics_in_clinical_input']
        if GENERAL.search(text):
            return False, ['cohort_or_general_statement']
        if TREATMENT.search(text):
            return False, ['treatment_content']
        return bool(RX[field].search(text)), [field + '_cue'] if RX[field].search(text) else ['no_field_cue']
    if field == 'imaging_description':
        if re.search(r'\b(?:clinical\s+diagnosis|diagnosed)\b', text, re.I) and not MRI.search(text):
            return False, ['clinical_diagnosis_not_image_observation']
        if CT.search(text) and not MRI.search(text):
            return False, ['ct_only_not_mri']
        if CT.search(text) and MRI.search(text):
            return False, ['mixed_ct_mri_sentence_needs_attribution']
        if TREATMENT.search(text):
            return False, ['treatment_content']
        observed = bool(re.search(r'\b(?:show\w*|reveal\w*|demonstrat\w*|display\w*|hyperintens\w*|hypointens\w*|signal|lesions?|infarct\w*|microbleeds?|lacunes?|atrophy)\b', text, re.I))
        eligible = bool(RX[field].search(text)) and observed
        return eligible, ['imaging_observation_cues'] if eligible else ['no_image_observation_cues']
    # Diagnostic search is test-result focused; existing author conclusions are reused separately.
    if TREATMENT.search(text):
        return False, ['treatment_content']
    if re.search(r'\b(?:offspring|chance\s+of|risk\s+of\s+inheriting|would\s+have|should\s+be|counsell\w*|recommended|prognos\w*)\b', text, re.I):
        return False, ['recommendation_or_inheritance_risk_not_test_result']
    if GENERAL.search(text):
        return False, ['cohort_or_general_statement']
    test = bool(RX['genetic_results'].search(text) or (field != 'genetic_results' and re.search(r'\b(?:test|analysis|screen|antibod\w*|laboratory)\b', text, re.I)))
    result = bool(RX['diagnostic_results'].search(text))
    return test and result, ['test_and_result_cues'] if test and result else ['no_explicit_test_result_cues']


def figure_numbers(text):
    numbers = set()
    for match in re.finditer(r'(?<!\w)(?:Figures?|Figs?\.?)\s+(\d+[A-Za-z]?(?:\s*(?:[-–,]|and|&)\s*\d+[A-Za-z]?)*)', text, re.I):
        raw = match[1]
        ranges = list(re.finditer(r'(\d+)[A-Za-z]?\s*[-–]\s*(\d+)[A-Za-z]?', raw))
        for r in ranges:
            a, b = map(int, r.groups())
            if 0 < b-a <= 30:
                numbers.update(range(a, b+1))
        numbers.update(map(int, re.findall(r'\d+', raw)))
    return numbers


def target_figure_number(fid):
    match = re.match(r'^(?:fig|f)(\d+)(?:$|[-_.])', fid, re.I)
    return int(match[1]) if match else None


def attach_figure_sentences(row, passages):
    target = target_figure_number(row['figure_id'])
    if target is None:
        return []
    found = []
    for p in passages:
        if p.get('section') in {'FIG', 'TITLE', 'ABSTRACT', 'INTRO', 'DISCUSS', 'CONCL', 'REF', 'AUTH_CONT', 'COMP_INT'}:
            continue
        base = {**p, 'passage_source_offset': p['source_offset'], 'start_in_passage': 0}
        for span in sentence_spans(base):
            refs = figure_numbers(span['text'])
            if target not in refs:
                continue
            allowed, reasons = route(span['text'], 'imaging_description')
            found.append({**span, 'figure_numbers': sorted(refs),
                          'routing_status': 'figure_reference_candidate' if allowed and refs == {target}
                          else 'ambiguous_or_non_mri_reference',
                          'routing_reasons': reasons + (['multiple_figures_in_same_sentence'] if len(refs) > 1 else []),
                          'image_pixels_verified': False})
    return found


def ground_existing(evidence, passages):
    mapping = {p['passage_id']: p for p in passages}
    output = []
    for ev in evidence:
        p = mapping.get(ev['passage_id'])
        quote = ev['quote']
        if p is None or not isinstance(quote, str) or not quote.strip() or quote not in p['text']:
            raise ValueError('Existing evidence quote no longer matches source')
        a = p['text'].index(quote)
        output.append({**ev, 'text': quote, 'start_in_passage': a, 'end_in_passage': a+len(quote),
                       'passage_source_offset': p['source_offset'], 'method': 'reused_model_quote_exact_match',
                       'semantic_validation': 'not_performed'})
    return output
