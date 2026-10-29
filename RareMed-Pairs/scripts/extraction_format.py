"""Deterministic formatting repair; no medical claim changes."""
import copy
import re


def fold_spaces(text):
    chars, spans = [], []
    for m in re.finditer(r'\s+|\S',text):
        chars.append(' ' if m.group().isspace() else m.group())
        spans.append((m.start(),m.end()))
    return ''.join(chars), spans


def source_quote(quote, text):
    if quote in text:
        return quote
    folded, spans = fold_spaces(text)
    target,_ = fold_spaces(quote)
    start = folded.find(target)
    if not target or start<0:
        return None
    return text[spans[start][0]:spans[start+len(target)-1][1]]


def normalize(response, request):
    candidate=copy.deepcopy(response)
    changes=[]
    if not isinstance(candidate,dict):
        return candidate,changes
    if set(candidate)=={'type','article_id','figures'} and candidate['type'] in ('object','json_object'):
        removed=candidate['type']
        del candidate['type']
        changes.append({'kind':'remove_redundant_root_type','original_value':removed})
    passages={p['passage_id']:p['text'] for p in request['passages']}
    for row in candidate.get('figures',[]):
        for role in ('link_evidence','diagnosis_evidence','basis_evidence'):
            for e in row.get(role,[]):
                original=e.get('quote')
                text=passages.get(e.get('passage_id'))
                if not isinstance(original,str) or not isinstance(text,str):
                    continue
                replacement=source_quote(original,text)
                if replacement is not None and replacement!=original:
                    changes.append({'kind':'restore_exact_source_whitespace','figure_id':row['figure_id'],
                        'evidence_role':role,'passage_id':e['passage_id'],
                        'original_quote':original,'restored_quote':replacement})
                    e['quote']=replacement
    return candidate,changes

