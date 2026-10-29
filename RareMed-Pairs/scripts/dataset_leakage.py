"""Checks for detected dataset overlap and answer leakage; not a privacy/deidentification tool."""
from collections import defaultdict
import re
import unicodedata

SPLITS = {'train', 'validation', 'test'}


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


class UnionFind:
    def __init__(self, ids):
        self.parent = {key: key for key in ids}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def value(row, key):
    return row.get(key) or row.get('metadata', {}).get(key)


def user_and_answer(row):
    users, answers = [], []
    for message in row.get('messages', []):
        content = message.get('content', [])
        text = content if isinstance(content, str) else '\n'.join(
            c.get('text', '') for c in content if isinstance(c, dict) and c.get('type') == 'text')
        if message.get('role') == 'user': users.append(text)
        if message.get('role') == 'assistant': answers.append(text)
    return '\n'.join(users), '\n'.join(answers)


def audit_records(records, relations=(), split_field='split', exposure=None):
    index = {}
    for row in records:
        sid = row['sample_id']
        if sid in index:
            raise ValueError('Repeated sample ID: ' + sid)
        index[sid] = row
    failures, warnings = [], []
    memberships = defaultdict(lambda: defaultdict(set))
    allocated = {}
    exposure = exposure or {}
    exposed_samples = set(exposure.get('sample_ids', []))
    exposed_articles = set(exposure.get('article_ids', []))
    exposed_groups = {value(r, 'split_group_id') for r in records if
                      r['sample_id'] in exposed_samples or value(r, 'article_id') in exposed_articles}
    exposed_groups.discard(None)
    def add(kind, key, sid, split):
        if key:
            memberships[(kind, str(key))][split].add(sid)
    for sid, row in index.items():
        split = value(row, split_field)
        if split in SPLITS:
            allocated[sid] = split
            if split != 'train' and (sid in exposed_samples or value(row, 'article_id') in exposed_articles or
                                     value(row, 'split_group_id') in exposed_groups):
                failures.append({'type': 'known_training_exposure_in_evaluation', 'sample_id': sid, 'split': split})
            add('source_article', value(row, 'article_id'), sid, split)
            add('split_group', value(row, 'split_group_id'), sid, split)
            add('provisional_case_within_article', value(row, 'provisional_case_key'), sid, split)
            # Unverified model labels must not be treated as global patient identities.
            if value(row, 'case_identity_verified'):
                add('verified_patient', value(row, 'verified_patient_id'), sid, split)
            for asset in row.get('dedup_asset_fingerprints', row.get('assets', [])):
                add('image_bytes', asset.get('sha256'), sid, split)
                add('decoded_pixels', asset.get('pixel_sha256'), sid, split)
            add('image_bytes', value(row, 'image_sha256'), sid, split)
            add('decoded_pixels', value(row, 'pixel_sha256'), sid, split)
        user, answer = user_and_answer(row)
        u, a = normalize(user), normalize(answer)
        if a and len(a) >= 30 and len(a.split()) >= 5 and a in u:
            failures.append({'type': 'verbatim_target_in_user_input', 'sample_id': sid})
        if a and split in SPLITS and len(a) >= 50 and len(a.split()) >= 8:
            add('long_exact_target_text', a, sid, split)
        # Context-dependent diagnostic clues are flagged, not automatically removed.
        if re.search(r'\b(?:diagnos(?:ed|is)\s+(?:with|of)|confirmed\s+CADASIL|pathogenic\s+NOTCH3|NOTCH3\s+(?:pathogenic\s+)?(?:variant|mutation))\b', user, re.I):
            warnings.append({'type': 'possible_diagnostic_answer_clue_in_input', 'sample_id': sid})
        if user and not row.get('messages'):
            warnings.append({'type': 'input_format_not_inspected', 'sample_id': sid})
        ancestors = value(row, 'parent_sample_ids') or []
        if isinstance(ancestors, str): ancestors = [ancestors]
        for ancestor in ancestors:
            if ancestor not in index:
                warnings.append({'type': 'lineage_parent_not_in_manifest', 'sample_id': sid, 'parent': ancestor})
            elif split in SPLITS:
                parent_split = value(index[ancestor], split_field)
                if parent_split in SPLITS and parent_split != split:
                    failures.append({'type': 'parent_derivative_cross_split', 'sample_id': sid, 'parent': ancestor})
    for (kind, key), by_split in memberships.items():
        if len(by_split) > 1:
            failures.append({'type': kind + '_cross_split', 'key': key,
                             'samples_by_split': {s: sorted(ids) for s, ids in by_split.items()}})
    for relation in relations:
        a, b = relation['sample_a'], relation['sample_b']
        if a in allocated and b in allocated and allocated[a] != allocated[b]:
            failures.append({'type': 'detected_relation_cross_split', 'sample_a': a, 'sample_b': b,
                             'relation_reasons': relation['reasons']})
    return {'records': len(index), 'allocated_records': len(allocated), 'split_field': split_field,
            'cross_split_check_performed': len(set(allocated.values())) >= 2,
            'message_records_checked': sum(bool(r.get('messages')) for r in records),
            'failures': failures, 'warnings': warnings,
            'status': 'failed' if failures else 'no_detected_leaks' if len(set(allocated.values())) >= 2 else 'split_check_not_applicable',
            'limitations': ['Only detected links and supported text formats are checked.',
                            'No guarantee of unseen-patient separation across different publications.',
                            'Diagnostic clues are task dependent; no clinical adjudication or PHI deidentification.']}
