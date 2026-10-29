"""Verify exact source slicing and local scope without loading model weights."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest
from types import SimpleNamespace
from medcpt_local import corpus, sha


class LocalRetrievalTests(unittest.TestCase):
    def test_invalid_article_id_rejected(self):
        with self.assertRaises(ValueError):
            corpus('../other')

    def test_smoke_fixture_is_marked_synthetic(self):
        self.assertTrue(all(r['passage_id'].startswith('synthetic_') for r in corpus(None)))

    def test_article_chunks_preserve_source_text_and_offsets(self):
        text = 'A B C D E'
        class Tokenizer:
            def __call__(self, value, **kwargs):
                return {'offset_mapping': [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]}
        article = {'passages': [{'text': text, 'passage_id': 'p0010',
                                'source_offset': 100, 'section': 'CASE'}]}
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()))
        with patch.object(Path, 'read_text', return_value=json.dumps(article)), \
             patch.dict('sys.modules', {'transformers': fake}):
            rows = corpus('PMC123')
        self.assertEqual(rows[0]['text'], text[rows[0]['start_in_passage']:rows[0]['end_in_passage']])
        self.assertEqual(rows[0]['passage_source_offset'], 100)
        self.assertEqual(rows[0]['article_id'], 'PMC123')

    def test_section_filter_does_not_include_discussion(self):
        class Tokenizer:
            def __call__(self, value, **kwargs):
                return {'offset_mapping': [(0, len(value))]}
        article = {'passages': [
            {'text': 'case evidence', 'passage_id': 'p1', 'source_offset': 0, 'section': 'CASE'},
            {'text': 'general discussion', 'passage_id': 'p2', 'source_offset': 20, 'section': 'DISCUSS'}]}
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()))
        with patch.object(Path, 'read_text', return_value=json.dumps(article)), \
             patch.dict('sys.modules', {'transformers': fake}):
            rows = corpus('PMC123', 'CASE')
        self.assertEqual([r['passage_id'] for r in rows], ['p1'])


if __name__ == '__main__':
    unittest.main()
