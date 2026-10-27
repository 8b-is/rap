"""
Unit tests for Panasonic Connect RAP (`rap_mem8_bfp64_pq.py`) Omni BFP-64
Contextual Memory Bank, Precomputed Integer Cosine Recall, and obs index parity fix.
"""

import importlib.util
import math
import os
import random
import sys
import unittest

_MOD_PATH = os.path.join(os.path.dirname(__file__), "..", "rap_mem8_bfp64_pq.py")
_SPEC = importlib.util.spec_from_file_location("rap_mem8_bfp64_pq", _MOD_PATH)
rap_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["rap_mem8_bfp64_pq"] = rap_mod
assert _SPEC.loader is not None
_SPEC.loader.exec_module(rap_mod)

Bfp64EmbeddingRowPq = rap_mod.Bfp64EmbeddingRowPq
PQ_Q30 = rap_mod.PQ_Q30
RapContextualMemoryBfp64Pq = rap_mod.RapContextualMemoryBfp64Pq


def _float_cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class TestPanasonicRapMem8Bfp64Pq(unittest.TestCase):
    def test_bfp64_compression_and_integer_cosine_accuracy(self):
        rng = random.Random(101)
        dim = 384  # sentence-transformers/all-MiniLM-L6-v2
        bank = RapContextualMemoryBfp64Pq(dim=dim)
        self.assertEqual(bank.compression_ratio_pq(), (256, 65))

        for _ in range(25):
            a = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            b = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            ra = Bfp64EmbeddingRowPq.from_floats(a)
            rb = Bfp64EmbeddingRowPq.from_floats(b)
            cos_int = ra.cosine_q30(rb) / float(PQ_Q30)
            cos_ref = _float_cosine(a, b)
            self.assertLess(abs(cos_int - cos_ref), 0.004)

    def test_zero_blocks_preserve_cosine_and_relative_scale(self):
        a = [1.0] * 64 + [2.0 ** -10] * 64
        b = [1.0] * 64 + [0.0] * 64
        for padding in ([], [0.0] * 1, [0.0] * 64, [0.0] * 128):
            with self.subTest(padding=len(padding)):
                ra = Bfp64EmbeddingRowPq.from_floats(a + padding)
                rb = Bfp64EmbeddingRowPq.from_floats(b + padding)
                self.assertAlmostEqual(ra.cosine_q30(rb) / PQ_Q30,
                                       _float_cosine(a, b), delta=1e-8)

    def test_wide_block_ranges_and_signed_cosine(self):
        # Powers of two are exactly representable by the block quantizer.
        for exponent in (-54, -30, 0, 30, 66):
            for tail in (1, 63, 64):
                a = [2.0 ** exponent] * 64 + [2.0 ** -54] * tail
                b = [2.0 ** exponent] * 64 + [-2.0 ** -54] * tail
                for sign in (-1, 1):
                    with self.subTest(exponent=exponent, tail=tail, sign=sign):
                        signed_b = [sign * x for x in b]
                        ra = Bfp64EmbeddingRowPq.from_floats(a)
                        rb = Bfp64EmbeddingRowPq.from_floats(signed_b)
                        score = ra.cosine_q30(rb)
                        self.assertLessEqual(abs(score), PQ_Q30)
                        self.assertEqual(score, rb.cosine_q30(ra))
                        self.assertAlmostEqual(score / PQ_Q30,
                                               _float_cosine(a, signed_b), delta=1e-8)
                        self.assertAlmostEqual(ra.cosine_q30(ra) / PQ_Q30,
                                               1.0, delta=1e-8)

    def test_zero_vectors_and_orthogonal_vectors(self):
        zero = Bfp64EmbeddingRowPq.from_floats([0.0] * 128)
        a = Bfp64EmbeddingRowPq.from_floats([1.0] * 64 + [0.0] * 64)
        b = Bfp64EmbeddingRowPq.from_floats([0.0] * 64 + [1.0] * 64)
        self.assertEqual(zero.cosine_q30(a), 0)
        self.assertEqual(a.cosine_q30(zero), 0)
        self.assertEqual(zero.cosine_q30(zero), 0)
        self.assertEqual(a.cosine_q30(b), 0)

    def test_obs_vs_act_index_parity_and_topk_recall(self):
        rng = random.Random(202)
        dim = 128
        bank = RapContextualMemoryBfp64Pq(dim=dim)

        target_obs = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        target_act = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        target_task = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        target_cat = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        target_plan = [rng.gauss(0.0, 1.0) for _ in range(dim)]

        # Episode 0 has target_act at index 2 (even) and target_obs at index 3 (odd)
        ep0_steps = [
            [rng.gauss(0.0, 1.0) for _ in range(dim)],  # 0: act_0
            [rng.gauss(0.0, 1.0) for _ in range(dim)],  # 1: obs_0
            target_act,                                 # 2: act_1
            target_obs,                                 # 3: obs_1
        ]
        bank.add_episode("ep_target", target_task, target_cat, target_plan, ep0_steps)

        for idx in range(5):
            bank.add_episode(
                f"ep_noise_{idx}",
                [rng.gauss(0.0, 1.0) for _ in range(dim)],
                [rng.gauss(0.0, 1.0) for _ in range(dim)],
                [rng.gauss(0.0, 1.0) for _ in range(dim)],
                [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(4)],
            )

        hits_act = bank.retrieve_topk(
            target_task, target_cat, target_plan, step_q=target_act, mode="act", k=2
        )
        self.assertEqual(hits_act[0][0], "ep_target")
        self.assertEqual(hits_act[0][1], 2)  # even action index

        hits_obs = bank.retrieve_topk(
            target_task, target_cat, target_plan, step_q=target_obs, mode="obs", k=2
        )
        self.assertEqual(hits_obs[0][0], "ep_target")
        self.assertEqual(hits_obs[0][1], 3)  # odd observation index (2*1 + 1)


if __name__ == "__main__":
    unittest.main()
