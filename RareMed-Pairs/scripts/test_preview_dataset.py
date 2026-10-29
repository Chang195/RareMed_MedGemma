import copy
import unittest
from preview_dataset import prepare_preview, classify_record, check_span, render_span


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.row = {'sample_id': 'PMC1_F1', 'caption': 'FLAIR image shows white matter lesions.',
            'prior_linkage': {'case_label': 'index patient', 'link_status': 'explicit', 'diagnosis_name': 'CADASIL',
                              'diagnosis_status': 'author_reported_confirmed'},
            'figure_reference_spans': [], 'author_diagnosis': {'name': 'CADASIL', 'status': 'author_reported_confirmed'}}
        self.pool = {'candidate_pools': {'presentation': [{'text': 'The patient had headache.', 'passage_id': 'p1',
                     'start_in_passage': 0, 'end_in_passage': 25}]}}

    def test_candidates_never_assigned_to_image_or_training(self):
        result = prepare_preview(self.row, self.pool)
        self.assertIsNone(result['clinical_context_assigned'])
        self.assertIsNone(result['task_drafts']['image_and_context_diagnosis']['input_clinical_text'])
        self.assertFalse(result['training_ready'])
        self.assertEqual(result['task_drafts']['image_and_context_diagnosis']['target_candidate'], 'CADASIL')

    def test_mouse_cadasil_not_human_diagnosis_task(self):
        self.row['prior_linkage']['case_label'] = 'CADASIL mouse model'
        result = prepare_preview(self.row, self.pool)
        self.assertIn('animal_model_text_signal', result['text_screening_flags'])
        self.assertIsNone(result['task_drafts']['image_and_context_diagnosis']['target_candidate'])

    def test_suspected_not_confirmed(self):
        self.row['prior_linkage']['diagnosis_status'] = 'suspected'
        result = prepare_preview(self.row, self.pool)
        self.assertEqual(result['preview_disposition'], 'single_case_uncertain_diagnosis')
        self.assertIsNone(result['task_drafts']['image_and_context_diagnosis']['target_candidate'])

    def test_diagram_is_held(self):
        self.row['caption'] = 'Flowchart of participants.'
        self.assertEqual(classify_record(self.row)[0], 'hold_nonpatient_or_unresolved')

    def test_caption_kept_verbatim_and_not_added_to_input(self):
        self.row['caption'] = 'MRI in a patient with CADASIL.'
        result = prepare_preview(self.row, self.pool)
        task = result['task_drafts']['image_description']
        self.assertEqual(task['target_candidate'], self.row['caption'])
        self.assertTrue(task['flags'])
        self.assertNotIn('CADASIL', task['question'])

    def test_invalid_source_slice_rejected(self):
        with self.assertRaises(ValueError):
            check_span({'passage_id': 'p1', 'start_in_passage': 0, 'end_in_passage': 4, 'text': 'fake'},
                       {'p1': {'text': 'real'}})

    def test_html_escapes_source_text(self):
        self.assertNotIn('<script>', render_span({'text': '<script>alert(1)</script>'}))


if __name__ == '__main__':
    unittest.main()
