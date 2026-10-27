"""Standalone BFP-64 episodic-memory prototype with integer cosine scoring.

The ALFWorld and WebShop benchmark scripts do not import this module; they use
sentence-transformer embeddings and their own retrieval functions. Related design
references include https://github.com/8b-is/far, https://github.com/8b-is/Engram,
and https://github.com/8b-is/alexiai. No integration with those projects is provided
by this module.

Rows quantize float inputs in blocks of at most 64 values, then store aligned
mantissas as Python integers with an adaptive-precision reciprocal norm. A packed
layout with one byte per mantissa and one byte per block exponent would use
390 bytes for 384 values, versus 1536 bytes for FP32. ``raw_block_bytes`` and
``compression_ratio_pq`` describe that hypothetical payload only; this module
neither serializes that layout nor measures Python object memory usage. Cached
norms, alignment metadata and Python container overhead are excluded.

``cosine_q30`` uses integer arithmetic on already quantized rows. Insertion and
query preparation through ``from_floats`` use floating-point quantization and an
integer square root to build each reciprocal norm. Quantization is lossy, and
cosine scores are approximations rather than exact cosines of the float inputs.

Episodes live in a process-local list with no persistence, update or deletion API.
``reward_pq`` is stored as a reduced rational but does not affect retrieval scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Dict, List, Optional, Sequence, Tuple

BFP_BLOCK_SIZE: int = 64
PQ_Q30: int = 1 << 30


def reduce_rational_pq(num: int, den: int) -> Tuple[int, int]:
    """Return exact GCD-reduced (p, q) with q > 0."""
    if den == 0:
        return (0, 1)
    if den < 0:
        num, den = -num, -den
    g = math.gcd(abs(num), den)
    return (num // g, den // g)


def isqrt_u128(n: int) -> int:
    """Exact integer floor square root floor(sqrt(n))."""
    if n < 2:
        return max(0, n)
    return math.isqrt(n)


@dataclass(frozen=True)
class Bfp64EmbeddingRowPq:
    """
    Omni BFP-64 embedding with an adaptive-precision reciprocal L2 norm.
    Since cosine similarity is scale-invariant, the shared block exponent cancels out
    when blocks are integer-aligned to `min_exp`.
    """

    aligned_mantissas: Tuple[int, ...]
    min_exp: int
    inv_norm_q30: int
    raw_block_bytes: int  # Hypothetical packed payload, not Python object size.
    # The reciprocal is inv_norm_q30 / 2**(30 + inv_norm_extra_bits).
    inv_norm_extra_bits: int = 0

    @classmethod
    def from_floats(cls, vec: Sequence[float]) -> "Bfp64EmbeddingRowPq":
        dim = len(vec)
        if dim == 0:
            raise ValueError("embedding must not be empty")
        if not all(math.isfinite(v) for v in vec):
            raise ValueError("embedding values must be finite")
        num_blocks = (dim + BFP_BLOCK_SIZE - 1) // BFP_BLOCK_SIZE
        block_mants: List[List[int]] = []
        block_exps: List[int] = []

        for b in range(num_blocks):
            blk = vec[b * BFP_BLOCK_SIZE : (b + 1) * BFP_BLOCK_SIZE]
            max_abs = max((abs(v) for v in blk), default=0.0)
            if max_abs < 1e-18:
                block_mants.append([0] * len(blk))
                block_exps.append(-60)
            else:
                e = max(-60, min(60, math.ceil(math.log2(max_abs / 127.0))))
                inv_s = math.ldexp(1.0, -e)
                m = [max(-127, min(127, int(round(v * inv_s)))) for v in blk]
                block_mants.append(m)
                block_exps.append(e)

        min_exp = min(block_exps)
        aligned: List[int] = []
        for m_blk, e in zip(block_mants, block_exps):
            # Python integers preserve the full (at most 120-bit) exponent gap.
            shift = e - min_exp
            mult = 1 << shift
            aligned.extend(v * mult for v in m_blk)

        sum_sq = sum(v * v for v in aligned)
        inv_norm_extra_bits = sum_sq.bit_length() // 2
        if sum_sq == 0:
            inv_norm_q30 = 0
        else:
            norm_q30 = max(1, isqrt_u128(sum_sq << 60))
            # Keep about 30 significant reciprocal bits even when alignment
            # makes the integer norm much larger than 2**30.
            inv_norm_q30 = ((1 << (60 + inv_norm_extra_bits)) + (norm_q30 >> 1)) // norm_q30

        return cls(
            aligned_mantissas=tuple(aligned),
            min_exp=min_exp,
            inv_norm_q30=inv_norm_q30,
            raw_block_bytes=dim + num_blocks,
            inv_norm_extra_bits=inv_norm_extra_bits,
        )

    def cosine_q30(self, query_row: "Bfp64EmbeddingRowPq") -> int:
        """
        Approximate cosine of the quantized vectors in Q30 [-2^30, +2^30].
        Scoring uses precomputed reciprocal norms without division or sqrt.
        """
        if len(self.aligned_mantissas) != len(query_row.aligned_mantissas):
            raise ValueError("cosine requires embeddings of equal dimension")
        if self.inv_norm_q30 == 0 or query_row.inv_norm_q30 == 0:
            return 0
        dot = sum(a * b for a, b in zip(self.aligned_mantissas, query_row.aligned_mantissas))
        prod = dot * self.inv_norm_q30 * query_row.inv_norm_q30
        shift = 30 + self.inv_norm_extra_bits + query_row.inv_norm_extra_bits
        magnitude = (abs(prod) + (1 << (shift - 1))) >> shift
        # Reciprocal rounding can put self-similarity a few units above Q30.
        magnitude = min(PQ_Q30, magnitude)
        return -magnitude if prod < 0 else magnitude


@dataclass
class RapEpisodeEntryPq:
    """One in-memory episode containing quantized Python row objects."""

    episode_id: str
    task_emb: Bfp64EmbeddingRowPq
    category_emb: Bfp64EmbeddingRowPq
    plan_emb: Bfp64EmbeddingRowPq
    action_obs_embs: List[Bfp64EmbeddingRowPq]  # alternating [act_0, obs_0, act_1, obs_1, ...]
    reward_pq: Tuple[int, int] = (1, 1)


class RapContextualMemoryBfp64Pq:
    """
    In-memory prototype accepting float vectors and ranking quantized rows.

    The benchmark scripts use separate retrieval implementations.
    """

    def __init__(self, dim: int = 384):
        if isinstance(dim, bool):
            raise TypeError("dim must be an integer, not a boolean")
        self.dim = operator.index(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self.episodes: List[RapEpisodeEntryPq] = []

    def _encode_vector(self, vec: Sequence[float], name: str) -> Bfp64EmbeddingRowPq:
        if len(vec) != self.dim:
            raise ValueError(f"{name} must have dimension {self.dim}, got {len(vec)}")
        return Bfp64EmbeddingRowPq.from_floats(vec)

    def add_episode(
        self,
        episode_id: str,
        task_vec: Sequence[float],
        category_vec: Sequence[float],
        plan_vec: Sequence[float],
        action_obs_vecs: Sequence[Sequence[float]],
        reward_pq: Tuple[int, int] = (1, 1),
    ) -> None:
        self.episodes.append(
            RapEpisodeEntryPq(
                episode_id=episode_id,
                task_emb=self._encode_vector(task_vec, "task_vec"),
                category_emb=self._encode_vector(category_vec, "category_vec"),
                plan_emb=self._encode_vector(plan_vec, "plan_vec"),
                action_obs_embs=[self._encode_vector(v, "action_obs_vecs") for v in action_obs_vecs],
                reward_pq=reduce_rational_pq(reward_pq[0], reward_pq[1]),
            )
        )

    def compression_ratio_pq(self) -> Tuple[int, int]:
        """Return FP32 / hypothetical packed-payload ratio, excluding metadata.

        This is a size calculation, not measured compression of Python objects.
        """
        num_blocks = (self.dim + BFP_BLOCK_SIZE - 1) // BFP_BLOCK_SIZE
        fp32_b = self.dim * 4
        bfp_b = self.dim + num_blocks
        return reduce_rational_pq(fp32_b, bfp_b)

    def retrieve_topk(
        self,
        task_q: Sequence[float],
        category_q: Sequence[float],
        plan_q: Optional[Sequence[float]] = None,
        step_q: Optional[Sequence[float]] = None,
        mode: str = "act",
        k: int = 3,
    ) -> List[Tuple[str, int, Tuple[int, int]]]:
        """
        Retrieve top-k episodes and matched step index `ret_index` with exact parity
        handling (`* 2` for `mode == 'act'`, `* 2 + 1` for `mode == 'obs'`).
        Returns list of `(episode_id, matched_step_index, score_pq_q30)`.
        Float queries are quantized and normalized before integer scoring;
        stored reward_pq values are not used in the score.
        """
        q_task = self._encode_vector(task_q, "task_q")
        q_cat = self._encode_vector(category_q, "category_q")
        q_plan = self._encode_vector(plan_q, "plan_q") if plan_q is not None else None
        q_step = self._encode_vector(step_q, "step_q") if step_q is not None else None

        scored: List[Tuple[int, str, int]] = []
        for ep in self.episodes:
            base_q30 = ep.task_emb.cosine_q30(q_task) + ep.category_emb.cosine_q30(q_cat)
            if q_plan is not None:
                base_q30 += ep.plan_emb.cosine_q30(q_plan)

            best_step_q30 = 0
            best_raw_idx = 0
            if q_step is not None and len(ep.action_obs_embs) > 0:
                offset = 1 if mode == "obs" else 0
                candidates = ep.action_obs_embs[offset::2]
                if candidates:
                    best_sub_idx = 0
                    best_step_q30 = candidates[0].cosine_q30(q_step)
                    for idx_c in range(1, len(candidates)):
                        sc = candidates[idx_c].cosine_q30(q_step)
                        if sc > best_step_q30:
                            best_step_q30 = sc
                            best_sub_idx = idx_c
                    best_raw_idx = best_sub_idx * 2 + offset

            total_q30 = base_q30 + best_step_q30
            scored.append((total_q30, ep.episode_id, best_raw_idx))

        scored.sort(key=lambda item: item[0], reverse=True)
        out: List[Tuple[str, int, Tuple[int, int]]] = []
        for sc_q30, ep_id, r_idx in scored[:k]:
            out.append((ep_id, r_idx, reduce_rational_pq(sc_q30, PQ_Q30)))
        return out
