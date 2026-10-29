import unittest
from screen import triage


class ScreeningChecks(unittest.TestCase):
    def test_missing_abstract_is_not_excluded(self):
        row = triage({'pmcid':'PMC1', 'title':'CADASIL MRI case report'})
        self.assertEqual(row['queue'], 'fulltext_required')
        self.assertFalse(row['training_ready'])

    def test_case_signals_have_source_evidence(self):
        row = triage({'pmcid':'PMC1', 'title':'CADASIL: a case report',
                      'abstractText':'MRI of the brain was performed.'})
        self.assertEqual(row['queue'], 'fulltext_required')
        self.assertEqual(row['evidence']['mri'][0]['field'], 'abstract')
        self.assertEqual(row['final_eligibility'], 'pending_fulltext_and_figure_review')

    def test_review_is_excluded(self):
        row = triage({'pmcid':'PMC1', 'title':'A CADASIL review', 'abstractText':'MRI features.'})
        self.assertEqual(row['queue'], 'excluded_review')

    def test_case_report_with_review_is_also_excluded(self):
        row=triage({'pmcid':'PMC1','title':'CADASIL MRI case report and literature review',
                    'abstractText':'MRI findings in a patient.'})
        self.assertEqual(row['queue'],'excluded_review')

    def test_no_title_or_abstract_disease_signal_still_goes_to_fulltext(self):
        row=triage({'pmcid':'PMC1','title':'Follow-up study','abstractText':'Clinical outcomes.'})
        self.assertEqual(row['queue'],'fulltext_required')


if __name__ == '__main__':
    unittest.main()
