from __future__ import annotations

import unittest

from scripts.check_code_quality import check_files, load_quality_config


class CodeQualityTests(unittest.TestCase):
    def test_quality_policy_is_loaded_from_project_config(self) -> None:
        config = load_quality_config()
        self.assertGreater(config.max_line_length, 0)
        self.assertIn("Str" + "Enum", config.forbidden_python310_tokens)

    def test_model_neutral_scripts_and_tests_pass_lightweight_quality_checks(self) -> None:
        issues = check_files()
        self.assertEqual([], [issue.format() for issue in issues])


if __name__ == "__main__":
    unittest.main()
