import unittest
from dataset_leakage import audit_records, UnionFind
from dedup_leakage import text_similarity, draft_split, phash, near_score


class LeakageTests(unittest.TestCase):
    def pair(self):
        return [{'sample_id': 'a', 'article_id': 'PMC1', 'split': 'train'},
                {'sample_id': 'b', 'article_id': 'PMC2', 'split': 'test'}]

    def test_same_article_cross_split_fails(self):
        records = self.pair(); records[1]['article_id'] = 'PMC1'
        self.assertEqual(audit_records(records)['status'], 'failed')

    def test_exact_pixels_cross_split_fails_even_when_bytes_differ(self):
        records = self.pair()
        for i, row in enumerate(records):
            row['dedup_asset_fingerprints'] = [{'sha256': str(i), 'pixel_sha256': 'samepixels'}]
        self.assertTrue(any(f['type'] == 'decoded_pixels_cross_split' for f in audit_records(records)['failures']))

    def test_near_candidate_relation_blocks_cross_split(self):
        report = audit_records(self.pair(), [{'sample_a': 'a', 'sample_b': 'b', 'reasons': ['perceptual_image_candidate']}])
        self.assertTrue(report['failures'])

    def test_unassigned_data_never_reported_as_split_pass(self):
        records = self.pair()
        for row in records: row['split'] = 'unassigned'
        report = audit_records(records)
        self.assertFalse(report['cross_split_check_performed'])
        self.assertEqual(report['status'], 'split_check_not_applicable')

    def test_answer_copy_in_user_fails(self):
        target = 'The MRI shows a large lesion in the right frontal lobe.'
        record = {'sample_id': 'a', 'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Describe: '+target}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': target}]}]}
        self.assertTrue(any(f['type'] == 'verbatim_target_in_user_input' for f in audit_records([record])['failures']))

    def test_parent_derivative_split_fails(self):
        records = self.pair(); records[1]['parent_sample_ids'] = ['a']
        self.assertTrue(any(f['type'] == 'parent_derivative_cross_split' for f in audit_records(records)['failures']))

    def test_model_case_label_is_not_global_patient_identity(self):
        records = self.pair()
        for row in records:
            row['verified_patient_id'] = 'index patient'
            row['case_identity_verified'] = False
        self.assertEqual(audit_records(records)['status'], 'no_detected_leaks')

    def test_verified_patient_cross_split_fails(self):
        records = self.pair()
        for row in records:
            row['verified_patient_id'] = 'curated-patient-1'
            row['case_identity_verified'] = True
        self.assertTrue(audit_records(records)['failures'])

    def test_smoke_article_not_allowed_as_test_even_without_exposed_image_present(self):
        records = [{'sample_id':'new_figure', 'article_id':'PMC1', 'split':'test'}]
        report = audit_records(records, exposure={'sample_ids':['old_figure'], 'article_ids':['PMC1']})
        self.assertTrue(any(f['type']=='known_training_exposure_in_evaluation' for f in report['failures']))

    def test_other_article_in_exposed_group_not_allowed_as_test(self):
        records = [{'sample_id':'a', 'article_id':'PMC1', 'split_group_id':'g', 'split':'hold'},
                   {'sample_id':'b', 'article_id':'PMC2', 'split_group_id':'g', 'split':'test'}]
        self.assertTrue(audit_records(records, exposure={'sample_ids':['a']})['failures'])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): audit_records([{'sample_id':'a'}, {'sample_id':'a'}])

    def test_grouping_and_split_are_deterministic(self):
        uf = UnionFind(['a','b','c']); uf.union('b','c'); uf.union('a','b')
        self.assertEqual({uf.find(k) for k in ['a','b','c']}, {'a'})
        self.assertEqual(draft_split('a', 42), draft_split('a', 42))

    def test_short_generic_caption_not_duplicate_evidence(self):
        self.assertEqual(text_similarity('Brain MRI.', 'Brain MRI.'), 0)

    def test_long_caption_normalization_keeps_negation(self):
        a = 'There is no evidence of acute infarction in the right frontal lobe on this image.'
        self.assertEqual(text_similarity(a, a.upper()), 1)
        self.assertLess(text_similarity(a, a.replace('no evidence of ', '')), 1)

    def test_mirror_hypothesis_detected_without_declaring_duplicate(self):
        from PIL import Image, ImageOps
        image = Image.new('L', (64, 64))
        image.putdata([(x*x+3*y+((x//8)%2)*40)%256 for y in range(64) for x in range(64)])
        mirror = ImageOps.mirror(image)
        a = {'phash63_variants': {'full': phash(image), 'mirror': phash(mirror)}}
        b = {'phash63_variants': {'full': phash(mirror)}}
        self.assertEqual(near_score(a, b)[0], 0)


if __name__ == '__main__':
    unittest.main()
