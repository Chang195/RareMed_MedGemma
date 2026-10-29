"""License declaration parsing; collection rules do not certify reuse rights."""
from html import unescape
import re


def collection_decision(fulltext, metadata_hint='', policy='exclude_by_nc_nd'):
    full=classify(fulltext)
    hint=classify(metadata_hint)
    uncertain={'unknown','unrecognized_declaration','CC_unspecified','conflicting_declarations'}
    selected=full if full['family'] not in uncertain else hint if hint['family'] not in uncertain else full
    if policy!='exclude_by_nc_nd':
        raise ValueError('Unknown collection license policy')
    excluded=any(x['family']=='CC BY-NC-ND' or 'CC BY-NC-ND' in x.get('detected',[]) for x in (full,hint))
    return {'collect':not excluded,'family':selected['family'],'version':selected['version'],
            'fulltext_family':full['family'],'metadata_family':hint['family'],
            'status':'excluded_by_nc_nd' if excluded else 'collection_only_rights_pending',
            'family_disagreement':full['family'] not in uncertain and hint['family'] not in uncertain and full['family']!=hint['family']}


def classify(text):
    t=unescape(text or '').lower()
    t=re.sub('[\u2010-\u2015\u2212]','-',t)
    t=re.sub(r'\s+',' ',t)
    # A CC0 waiver applying to attached data is not the article's CC BY license.
    if 'creativecommons.org/licenses/by' in t and 'public domain dedication waiver' in t:
        t=re.split(r'(?:the )?creative commons public domain dedication waiver',t,maxsplit=1)[0]
    if not t.strip():
        return {'family':'unknown','version':None}
    if re.search(r'\bcc\s*0\b|creativecommons\.org/publicdomain/zero',t):
        return {'family':'CC0','version':'1.0' if '1.0' in t else None}
    if 'public domain' in t or 'creativecommons.org/publicdomain/mark' in t:
        return {'family':'public_domain_statement','version':None}
    urls=re.findall(r'creativecommons\.org/licenses/(by(?:-(?:nc|nd|sa))*)/([1-4]\.0)',t)
    short=re.findall(r'\bcc\s*-?\s*by((?:\s*-?\s*(?:nc|nd|sa))*)',t)
    families=set()
    for name,_ in urls:
        families.add('CC '+name.upper())
    for suffix in short:
        codes=re.findall(r'nc|nd|sa',suffix)
        families.add('CC BY'+''.join('-'+c.upper() for c in codes))
    if not families and 'commons' in t and 'attribution' in t:
        nc=bool(re.search(r'non[ -]?commercial',t))
        nd=bool(re.search(r'no[ -]?(?:derivatives?|derivs?)',t))
        sa=bool(re.search(r'share[ -]?alike',t))
        families.add('CC BY'+('-NC' if nc else '')+('-ND' if nd else '-SA' if sa else ''))
    if len(families)>1:
        return {'family':'conflicting_declarations','version':None,'detected':sorted(families)}
    if families:
        versions={v for _,v in urls} or set(re.findall(r'\b[1-4]\.0\b',t))
        return {'family':next(iter(families)),'version':next(iter(versions)) if len(versions)==1 else None}
    if 'all rights reserved' in t:
        return {'family':'all_rights_reserved','version':None}
    if 'creative commons' in t:
        return {'family':'CC_unspecified','version':None}
    if 'research re-use' in t or 'research reuse' in t or ('pmc open access subset' in t and 're-use' in t):
        return {'family':'publisher_specific_terms','version':None}
    return {'family':'unrecognized_declaration','version':None}

