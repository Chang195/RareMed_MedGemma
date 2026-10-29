import unittest
from screen_fulltext import assess


def passage(section,text):
    return {'metadata':{'section_type':section},'text':text,'offset':100}


class FulltextChecks(unittest.TestCase):
    def test_body_can_supply_signals_missing_in_abstract(self):
        q,e=assess([passage('ABSTRACT','Clinical follow-up.'),passage('CASE','CADASIL was considered. MRI was performed.')])
        self.assertEqual(q,'fulltext_passed')
        self.assertEqual(e['mri'][0]['section'],'CASE')

    def test_references_alone_do_not_supply_disease(self):
        q,e=assess([passage('CASE','MRI was performed.'),passage('REF','CADASIL MRI review')])
        self.assertEqual(q,'excluded_fulltext_signals_insufficient')
        self.assertFalse(e['cadasil'])

    def test_abstract_only_response_held_not_excluded(self):
        q,_=assess([passage('ABSTRACT','CADASIL MRI')])
        self.assertEqual(q,'held_fulltext_incomplete')


if __name__=='__main__':
    unittest.main()
