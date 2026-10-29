import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch
import pipeline_runtime_repair as repair


class RuntimeTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parent / 'original-pipeline-new-test-v2/scripts/acquire_screened.py').exists(),
                         'Optional local-run integration fixture is not distributed')
    def test_original_acquisition_resumes_candidate_and_access_hold_without_network(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'original-pipeline-new-test-v2/scripts'))
        import acquire_screened as acquisition
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'data/manifests').mkdir(parents=True)
            (root / 'data/processed').mkdir()
            (root / 'reports').mkdir()
            source = root / 'data/manifests/input.jsonl'
            source.write_text(''.join(json.dumps({'article_id': aid, 'queue': 'fulltext_passed'})+'\n'
                                      for aid in ['PMC1', 'PMC2']), encoding='utf8')
            cfg = {'run_id': 'fixture', 'screened_manifest': 'data/manifests/input.jsonl',
                   'queues': ['fulltext_passed'], 'max_articles': 2, 'license_policy': 'exclude_by_nc_nd'}
            config = root / 'config.json'
            repair.atomic_dump(config, cfg)
            repair.atomic_dump(root / 'data/manifests/fixture_checkpoint.json', {
                'config': cfg, 'manifest_sha256': acquisition.digest(source), 'exclusion_hashes': {},
                'completed': {
                    'PMC1': {'status': {'article_id': 'PMC1', 'status': 'candidate'}, 'figures': []},
                    'PMC2': {'status': {'article_id': 'PMC2', 'status': 'held_access_challenge'}, 'figures': []}}})
            with patch.object(acquisition, 'ROOT', root), \
                 patch.object(acquisition, 'collect_article') as fetch, \
                 patch('sys.argv', ['acquire_screened.py', '--config', str(config)]):
                acquisition.main()
                fetch.assert_not_called()
            receipt = json.loads((root / 'reports/fixture_receipt.json').read_text(encoding='utf8'))
            self.assertEqual([r['status'] for r in receipt['results']], ['candidate', 'held_access_challenge'])

    def test_transient_replace_failure_recovers(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'checkpoint.json'
            real = repair.os.replace
            calls = []

            def transient(src, dst):
                calls.append(src)
                if len(calls) <= 2:
                    raise PermissionError('simulated sharing lock')
                return real(src, dst)

            with patch.object(repair.os, 'replace', side_effect=transient), patch.object(repair.time, 'sleep'):
                repair.atomic_dump(target, {'completed': 144})
            self.assertEqual(json.loads(target.read_text()), {'completed': 144})
            self.assertEqual(len(calls), 3)
            self.assertEqual(list(Path(td).glob('*.tmp')), [])

    def test_persistent_failure_preserves_previous_and_recovery_file(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'checkpoint.json'
            target.write_text('{"completed":143}')
            with patch.object(repair.os, 'replace', side_effect=PermissionError('lock')), patch.object(repair.time, 'sleep'):
                with self.assertRaisesRegex(PermissionError, 'recovery file retained'):
                    repair.atomic_dump(target, {'completed': 144})
            self.assertEqual(json.loads(target.read_text())['completed'], 143)
            recovery = list(Path(td).glob('*.tmp'))
            self.assertEqual(len(recovery), 1)
            self.assertEqual(json.loads(recovery[0].read_text())['completed'], 144)

    def test_challenge_is_access_hold_not_medical_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'article.html'
            p.write_text('<title>Checking your browser - reCAPTCHA</title>')
            self.assertTrue(repair.is_challenge(p))
            p.write_text('<title>Real article</title><p>mentions CAPTCHA in body</p>')
            self.assertFalse(repair.is_challenge(p))


if __name__ == '__main__':
    unittest.main()
