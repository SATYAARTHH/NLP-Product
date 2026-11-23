"""Focused checks for privacy masking and fixed evaluation definitions."""

import unittest

import numpy as np

from triageguard.modeling.metrics import expected_calibration_error, field_instances
from triageguard.modeling.redaction import Redactor


class RedactionTest(unittest.TestCase):
    def test_masks_full_identifiers(self):
        text = "Patient: Jane Smith   DOB: 04/02/1980   SSN: 999-12-3456\nPhone: 212-555-0112"
        safe, spans = Redactor().redact(text)
        self.assertNotIn("Jane Smith", safe)
        self.assertNotIn("999-12-3456", safe)
        self.assertEqual({span["label"] for span in spans}, {"NAME", "DATE_OF_BIRTH", "SSN", "PHONE"})


class MetricTest(unittest.TestCase):
    def test_perfect_probabilities_have_zero_ece(self):
        probabilities = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(expected_calibration_error(["a", "b"], probabilities, ["a", "b"]), 0.0)

    def test_list_elements_are_independent_instances(self):
        instances = field_instances({"procedures": ["A", "B"], "amount": 12.5})
        self.assertEqual(sum(instances.values()), 3)


if __name__ == "__main__":
    unittest.main()
