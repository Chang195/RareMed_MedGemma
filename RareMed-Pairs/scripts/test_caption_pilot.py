from pathlib import Path
import unittest
from build_caption_pilot import target_span, make_messages, within


class CaptionPilotTests(unittest.TestCase):
    def setUp(self):
        self.row = {'caption': 'Axial MRI shows an infarct (red arrow).',
                    'figure_id': 'F1', 'prior_linkage': {'figure_id': 'F1'}}
        self.article = {'passages': [{'passage_id': 'p1', 'source_offset': 50,
                                     'text': self.row['caption']}]}

    def test_target_is_exact_caption_slice(self):
        selection = {'target_text': 'an infarct (red arrow)', 'target_passage_id': 'p1'}
        evidence = target_span(selection, self.row, self.article)
        self.assertEqual(self.article['passages'][0]['text'][evidence['start_in_passage']:evidence['end_in_passage']], selection['target_text'])

    def test_paraphrase_is_rejected(self):
        with self.assertRaises(ValueError):
            target_span({'target_text': 'acute cerebral ischemia', 'target_passage_id': 'p1'}, self.row, self.article)

    def test_other_figure_identity_rejected(self):
        self.row['prior_linkage']['figure_id'] = 'F2'
        with self.assertRaises(ValueError):
            target_span({'target_text': self.row['caption'], 'target_passage_id': 'p1'}, self.row, self.article)

    def test_target_not_put_in_user_input(self):
        messages = make_messages('Describe the image.', 'An infarct (red arrow).')
        self.assertEqual(messages[0]['content'], [{'type': 'image'}, {'type': 'text', 'text': 'Describe the image.'}])
        self.assertEqual(messages[1]['role'], 'assistant')

    def test_dataset_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            within(Path('C:/example/dataset'), '../secret')


if __name__ == '__main__':
    unittest.main()
