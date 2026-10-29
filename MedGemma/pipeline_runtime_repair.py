"""Local filesystem recovery and access-challenge recognition. No access bypass."""
import json
import os
from pathlib import Path
import re
import tempfile
import time


def atomic_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    with os.fdopen(fd, 'w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(9):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 8:
                raise PermissionError('Checkpoint replacement remained blocked. Complete recovery file retained: '+str(temporary))
            time.sleep(min(.1 * 2**attempt, 1.5))


class AccessChallengeError(RuntimeError):
    pass


def is_challenge(path):
    path = Path(path)
    if path.suffix.lower() != '.html':
        return False
    text = path.read_text(encoding='utf8', errors='replace')
    title = re.search(r'<title\b[^>]*>(.*?)</title>', text, re.I | re.S)
    return bool(title and re.search(r'checking your browser|captcha|access denied|just a moment', title[1], re.I))


def apply_runtime_patch(raw):
    text = raw.decode('utf8').replace('\r\n', '\n')
    start = text.index('def dump(path, value):')
    end = text.index('\n\nclass Downloader:', start)
    text = text[:start] + '''from pipeline_runtime_repair import atomic_dump, AccessChallengeError, is_challenge


def dump(path, value):
    return atomic_dump(path, value)
''' + text[end:]
    anchor = "        self.events.append({'path': path.relative_to(ROOT).as_posix(), 'url': url,"
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''        if is_challenge(path):
            raise AccessChallengeError('Access challenge cached; held without retry or bypass: '+str(path))
''' + anchor)
    anchor = "    html = dl.get(f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/', raw / 'article.html')"
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    try:
        html = dl.get(f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/', raw / 'article.html')
    except AccessChallengeError as exc:
        return [], {'article_id': pmcid, 'status': 'held_access_challenge',
                    'reason': str(exc), 'retry_policy': 'manual_review_required_no_automatic_bypass',
                    'training_ready': False}
'''.rstrip())
    return text.encode('utf8')
