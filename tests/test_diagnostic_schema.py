from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class DiagnosticSchemaTests(unittest.TestCase):
    def test_inventory_actions_require_an_explicit_compatible_alias(self):
        from models.common.studies.sampling_audit_20261026.human_reference_comparison import is_inventory_exclusion
        current='exclude_from_diagnostic_inventory'
        aliases={'exclude_from_legacy_inventory':current}
        self.assertTrue(is_inventory_exclusion(current,aliases))
        self.assertTrue(is_inventory_exclusion('exclude_from_legacy_inventory',aliases))
        self.assertFalse(is_inventory_exclusion('exclude_from_human_inventory',aliases))
        self.assertFalse(is_inventory_exclusion('exclude_from_unknown_inventory',aliases))
        self.assertFalse(is_inventory_exclusion('include_in_diagnostic_inventory',aliases))


if __name__=='__main__':
    unittest.main()
