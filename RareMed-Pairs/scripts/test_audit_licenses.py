import unittest
from audit_licenses import classify


class LicenseAuditChecks(unittest.TestCase):
    def test_publisher_specific_permission_is_not_cc(self):
        self.assertEqual(classify('PMC Open Access Subset for unrestricted re-use, except commercial resale')['family'],'publisher_specific_terms')

    def test_article_by_and_data_cc0_remain_separate(self):
        text='Article: http://creativecommons.org/licenses/by/4.0/. The Creative Commons Public Domain Dedication waiver (http://creativecommons.org/publicdomain/zero/1.0/) applies to data.'
        self.assertEqual(classify(text),{'family':'CC BY','version':'4.0'})

    def test_unversioned_metadata_is_not_silently_version_four(self):
        self.assertEqual(classify('cc by'),{'family':'CC BY','version':None})

    def test_version_and_conditions_from_url(self):
        self.assertEqual(classify('https://creativecommons.org/licenses/by-nc-nd/3.0/'),{'family':'CC BY-NC-ND','version':'3.0'})

    def test_long_form_hyphens(self):
        self.assertEqual(classify('Creative Commons Attribution–NonCommercial–NoDerivatives 4.0')['family'],'CC BY-NC-ND')

    def test_generic_open_access_is_not_a_license(self):
        self.assertEqual(classify('open access')['family'],'unrecognized_declaration')

    def test_conflicting_declarations_held(self):
        self.assertEqual(classify('CC BY 4.0 and CC BY-NC 4.0')['family'],'conflicting_declarations')


if __name__=='__main__':
    unittest.main()
