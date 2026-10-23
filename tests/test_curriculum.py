"""Checks for batch boundaries and partial cross-cluster mixing."""

import unittest

import numpy as np
import pandas as pd

from saerl.curriculum.order import build_order
from saerl.difficulty.model import fit_prediction_table


def prediction_table(sizes):
    rows = []
    for cluster_id, count in enumerate(sizes):
        for position in range(count):
            rows.append({
                "sample_id": len(rows), "cluster_id": cluster_id,
                "ranking_score": position / count, "full_len_tok": 100 + position,
            })
    return pd.DataFrame(rows)


class CurriculumTests(unittest.TestCase):
    def test_uneven_clusters_preserve_batch_boundaries_and_coverage(self):
        source = prediction_table([200, 1000, 270])
        order = build_order(source, mix_tail=0)
        originals = order[~order.is_padding_copy]
        self.assertEqual(sorted(originals.sample_id), sorted(source.sample_id))
        for _, batch in order.groupby("batch_id"):
            self.assertEqual(len(batch), 128)
            self.assertEqual(batch.cluster_id.nunique(), 1)
        for _, group in originals.groupby("cluster_id"):
            self.assertTrue(group.ranking_score.is_monotonic_increasing)

    def test_mix8_exchanges_only_paired_tails(self):
        source = prediction_table([256, 256, 256])
        base = build_order(source, mix_tail=0)
        mixed = build_order(source, mix_tail=8)
        for batch_id, batch in mixed.groupby("batch_id"):
            before = base[base.batch_id == batch_id]
            self.assertEqual(batch.sample_id.iloc[:120].tolist(), before.sample_id.iloc[:120].tolist())
            partner = int(batch.partner_batch_id.iloc[0])
            if partner < 0:
                self.assertEqual(batch.sample_id.tolist(), before.sample_id.tolist())
                continue
            self.assertEqual(
                batch.sample_id.iloc[-8:].tolist(),
                base[base.batch_id == partner].sample_id.iloc[-8:].tolist(),
            )
            self.assertEqual(int(mixed[mixed.batch_id == partner].partner_batch_id.iloc[0]), batch_id)
            self.assertEqual(sorted(batch.cluster_id.value_counts().tolist()), [8, 120])
        self.assertEqual(sorted(mixed.sample_id), sorted(base.sample_id))
        self.assertTrue((mixed.partner_batch_id >= 0).any())

    def test_padded_batches_are_not_mixed(self):
        order = build_order(prediction_table([129, 256]), mix_tail=8)
        for _, batch in order.groupby("batch_id"):
            if batch.is_padding_copy.any():
                self.assertTrue((batch.partner_batch_id == -1).all())

    def test_tiny_tail_redistribution_and_cluster_padding(self):
        order = build_order(prediction_table([129]), mix_tail=0)
        counts = order.groupby("source_batch_id").is_padding_copy.agg(["count", "sum"])
        self.assertEqual((counts["count"] - counts["sum"]).tolist(), [65, 64])
        self.assertEqual(counts["count"].tolist(), [128, 128])
        self.assertEqual(order.cluster_id.nunique(), 1)
        tiny = build_order(prediction_table([1, 2]), batch_size=8, mix_tail=1)
        self.assertEqual(len(tiny), 16)
        self.assertEqual(tiny.loc[~tiny.is_padding_copy, "sample_id"].nunique(), 3)

    def test_reproducibility_and_pair_constraints(self):
        source = prediction_table([901, 1500, 2700, 3300])
        first = build_order(source)
        second = build_order(source.sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(first, second)
        batches = {key: batch for key, batch in first.groupby("batch_id")}
        for batch_id, batch in batches.items():
            partner = int(batch.partner_batch_id.iloc[0])
            if partner < 0:
                continue
            other = batches[partner]
            self.assertLessEqual(abs(batch_id - partner), 16)
            self.assertNotEqual(batch.base_cluster_id.iloc[0], other.base_cluster_id.iloc[0])
            self.assertLessEqual(abs(batch.stage.iloc[0] - other.stage.iloc[0]), 1)
            self.assertFalse(batch.is_padding_copy.any())
        same_cluster = build_order(prediction_table([512]))
        self.assertTrue((same_cluster.partner_batch_id == -1).all())

    def test_invalid_inputs_fail(self):
        source = prediction_table([2, 2])
        for kwargs in [{"batch_size": 0}, {"mix_tail": -1}, {"mix_tail": 129}, {"window_batches": -1}]:
            with self.assertRaises(ValueError):
                build_order(source, **kwargs)
        with self.assertRaises(ValueError):
            build_order(source.iloc[:0])
        with self.assertRaises(ValueError):
            build_order(pd.concat([source, source.iloc[:1]]))

    def test_saved_cluster_preprocessing_and_supervision_boundary(self):
        rng = np.random.default_rng(0)
        source = pd.DataFrame({
            "sample_id": np.arange(80), "prompt_len_tok": 10,
            "sol_len_tok": 20, "full_len_tok": 30,
            "f_a": rng.normal(size=80), "f_b": rng.normal(size=80),
            "difficulty_true": rng.normal(size=80),
        })
        table, fitted, _ = fit_prediction_table(
            source, backend="elasticnet", labeled_samples=20, n_clusters=3,
        )
        transformed = fitted["cluster_scaler"].transform(
            fitted["imputer"].transform(source[["f_a", "f_b"]].to_numpy())
        )
        np.testing.assert_array_equal(fitted["clusterer"].predict(transformed), table.cluster_id)
        source.loc[~table.has_difficulty_label, "difficulty_true"] = 100000
        perturbed, _, _ = fit_prediction_table(
            source, backend="elasticnet", labeled_samples=20, n_clusters=3,
        )
        np.testing.assert_allclose(table.ranking_score, perturbed.ranking_score)


if __name__ == "__main__":
    unittest.main()
