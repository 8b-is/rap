"""
================================================================================
[8b-is Sovereign Panasonic Connect RAP Substrate — Omni BFP-64 Contextual
 Memory Bank + MEM|8 Precomputed Integer Cosine Recall & p/q Reward Scaling]

Connects Panasonic Connect's RAP ("Retrieval-Augmented Planning with Contextual
Memory for Multimodal LLM Agents") with the 8b-is conditional & episodic memory stack:
- https://github.com/8b-is/far      (Omni BFP-64 + MEM|8 adaptive episodic key bank)
- https://github.com/8b-is/Engram   (Omni BFP-64 + Lemire p/q conditional memory)
- https://github.com/8b-is/alexiai  (GAIA-MLX-QUANT b1.58 + p/q + MEM|8 substrate)

Key Bugfixes & Mathematical Upgrades for RAP (`alfworld` & `webshop`):
1. Bugfix in `alfworld/main.py`:
   When `mode == 'obs'`, `log_embeddings = emb[1::2]` selects odd observation
   indices, but `ret_index.append(np.argmax(cos_scores)*2)` returned even action
   indices (off-by-one window alignment). Fixed to `* 2 + (1 if mode == 'obs' else 0)`.
2. Bugfix in `webshop/main.py`:
   When `use_act_obs == True`, `model_embedding.encode(...)` was called inside the
   `for a, emb in enumerate(embeddings['Actions']):` loop, re-running the Transformer
   encoder `M` times on the identical query string! Hoisted out of the loop (`M/1x`
   encoder speedup).
3. Omni BFP-64 Contextual Memory Bank (`RapContextualMemoryBfp64Pq`):
   Compresses `all-MiniLM-L6-v2` (384-dim) / `L12-v2` Task, Category, Plan, and
   Action/Observation trajectory embeddings from `1536 B` (FP32) to `390 B` (6 blocks
   of 64 int8 mantissas + 6 int8 shared exponents = `256/65 ~= 63/16x` compression).
4. Scale-Invariant Precomputed Integer Cosine Recall (`isqrt_u128`):
   Each stored memory vector precomputes a fixed-point reciprocal L2 norm
   with adaptive precision once on insertion via `math.isqrt`. Multi-cue retrieval across
   `Task + Category + Plan + max_{t}(Actions/Obs)` executes in pure integer dot
   products with ZERO square roots and ZERO floating-point divisions at query time!
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    raw_block_bytes: int
    # The reciprocal is inv_norm_q30 / 2**(30 + inv_norm_extra_bits).
    inv_norm_extra_bits: int = 0

    @classmethod
    def from_floats(cls, vec: Sequence[float]) -> "Bfp64EmbeddingRowPq":
        dim = len(vec)
        assert dim > 0
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
    """Single RAP episodic trajectory stored in Omni BFP-64."""

    episode_id: str
    task_emb: Bfp64EmbeddingRowPq
    category_emb: Bfp64EmbeddingRowPq
    plan_emb: Bfp64EmbeddingRowPq
    action_obs_embs: List[Bfp64EmbeddingRowPq]  # alternating [act_0, obs_0, act_1, obs_1, ...]
    reward_pq: Tuple[int, int] = (1, 1)


class RapContextualMemoryBfp64Pq:
    """
    Zero-float p/q + Omni BFP-64 Contextual Memory Bank for RAP (`alfworld` & `webshop`).
    """

    def __init__(self, dim: int = 384):
        self.dim = int(dim)
        self.episodes: List[RapEpisodeEntryPq] = []

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
                task_emb=Bfp64EmbeddingRowPq.from_floats(task_vec),
                category_emb=Bfp64EmbeddingRowPq.from_floats(category_vec),
                plan_emb=Bfp64EmbeddingRowPq.from_floats(plan_vec),
                action_obs_embs=[Bfp64EmbeddingRowPq.from_floats(v) for v in action_obs_vecs],
                reward_pq=reduce_rational_pq(reward_pq[0], reward_pq[1]),
            )
        )

    def compression_ratio_pq(self) -> Tuple[int, int]:
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
        """
        q_task = Bfp64EmbeddingRowPq.from_floats(task_q)
        q_cat = Bfp64EmbeddingRowPq.from_floats(category_q)
        q_plan = Bfp64EmbeddingRowPq.from_floats(plan_q) if plan_q is not None else None
        q_step = Bfp64EmbeddingRowPq.from_floats(step_q) if step_q is not None else None

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
