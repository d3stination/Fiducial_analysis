from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.final_analysis import participant_balanced_limits
from src.statistics.condition_morphology import permutation_p
from src.variability_analysis.bx_candidates import DEFINITIONS, DETECTORS


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"


class ReleaseTests(unittest.TestCase):
    def test_frozen_catalogue(self):
        registry = pd.read_csv(ROOT / "configs/canonical_method_registry.csv")
        self.assertEqual(registry.candidate_type.value_counts().to_dict(), {"X": 13, "B": 10})
        self.assertEqual(registry.algorithm_id.tolist(), [f"B{i}" for i in range(1, 11)] + [f"X{i}" for i in range(1, 14)])
        implemented = [definition for definition in DEFINITIONS if definition.implementation_status.startswith("IMPLEMENTED")]
        self.assertEqual(len(implemented), 23)
        self.assertEqual({definition.name for definition in implemented}, set(registry.candidate_name))
        self.assertEqual(len(DEFINITIONS) - len(implemented), 3)
        self.assertTrue(set(registry.candidate_name).issubset(DETECTORS))

    def test_frozen_source_hashes(self):
        expected = {
            "bx_candidates.py": "b5581c121542731b4b34ce8b33eb4b420ce33b43c93b02cf3ef4a97116a057cd",
            "phase4.py": "2351684a1c03a8a53fb66f4da1ea4fbb25c162aefad392212bfc5333f5099140",
            "signal_processing.py": "0a5b6741e50ec47773b4a8367825b428592a80c076a37bcf1ccb40a49ceaab2e",
        }
        for name, digest in expected.items():
            path = ROOT / "src/variability_analysis" / name
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), digest)

    def test_dataset_algorithm_reconciliation(self):
        table = pd.read_csv(TABLES / "dataset_algorithm_results.csv")
        self.assertEqual(len(table), 115)
        self.assertFalse(table.duplicated(["dataset", "condition", "point_type", "algorithm_id"]).any())
        self.assertTrue((table.total_beat_n == table.attempted_beat_n + table.upstream_ineligible_n).all())
        self.assertTrue((table.attempted_beat_n == table.valid_detection_n + table.missed_detection_n + table.invalid_detection_n).all())

    def test_reference_types_remain_distinct(self):
        table = pd.read_csv(TABLES / "dataset_algorithm_results.csv")
        self.assertEqual(set(table.loc[table.dataset.eq("HeartCycle"), "reference_type"]), {"ECHO AVO", "ECHO AVC"})
        self.assertTrue(table.loc[table.dataset.eq("IITB"), "reference_type"].str.contains("no independent", case=False).all())
        self.assertTrue(table.loc[table.dataset.eq("ReBeat"), "reference_type"].str.contains("Expert ICG").all())

    def test_iitb_x3_exact_sparse_permutation(self):
        table = pd.read_csv(TABLES / "condition_morphology_associations.csv")
        row = table.loc[
            table.dataset.eq("IITB") & table.point_type.eq("X") & table.algorithm_id.eq("X3")
            & table.outcome.eq("failure_rate_pct") & table.feature.eq("beat_to_beat_variability")
        ].iloc[0]
        self.assertAlmostEqual(row.spearman_rho, 0.3973597071195132)
        self.assertAlmostEqual(row.permutation_p, 2 / 18)
        self.assertAlmostEqual(row.bh_q_value, 0.9604357211337692)
        self.assertFalse(bool(row.exploratory_supported))
        x = np.arange(18, dtype=float)
        y = np.r_[np.zeros(17), 1.0]
        observed = 0.3973597071195132
        self.assertAlmostEqual(permutation_p(x, y, observed), 2 / 18)

    def test_condition_morphology_family(self):
        table = pd.read_csv(TABLES / "condition_morphology_associations.csv")
        self.assertEqual(len(table), 672)
        self.assertEqual(int(table.exploratory_supported.sum()), 14)

    def test_complete_morphology_family(self):
        table = pd.read_csv(TABLES / "morphology_associations.csv")
        self.assertEqual(len(table), 1794)
        self.assertEqual(table.feature.nunique(), 6)
        self.assertEqual(table.algorithm_id.nunique(), 23)

    def test_failure_reason_reconciliation(self):
        table = pd.read_csv(TABLES / "detection_failure_mechanisms.csv")
        group = ["dataset", "condition", "point_type", "algorithm_id"]
        self.assertEqual(table.groupby(group).ngroups, 115)
        self.assertTrue(table.groupby(group).reason_record_n.sum().eq(table.groupby(group).failed_or_ineligible_n.first()).all())

    def test_figure_package(self):
        figures = list((ROOT / "results/figures").rglob("*.png"))
        self.assertEqual(len(figures), 66)
        for figure in figures:
            for suffix in [".pdf", ".csv", ".md"]:
                self.assertTrue(figure.with_suffix(suffix).exists(), figure.name)

    def test_participant_balanced_limits(self):
        frame = pd.DataFrame({"subject_id": ["a", "a", "b", "b"], "difference_ms": [1.0, 3.0, 5.0, 7.0]})
        bias, low, high = participant_balanced_limits(frame)
        self.assertAlmostEqual(bias, 4.0)
        self.assertLess(low, bias)
        self.assertGreater(high, bias)


if __name__ == "__main__":
    unittest.main()
