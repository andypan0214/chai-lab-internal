"""Read one finished Chai-1 run into normalized candidate objects.

Files read (all written by chai_lab or run_chai.py, never re-computed here):

  pred.model_idx_<i>.cif        chai_lab run_folding_on_context: structure,
                                with per-atom pLDDT x 100 in B_iso_or_equiv
                                (chai_lab/chai1.py, "use 0-100 scale for
                                pLDDT in pdb outputs" -> save_to_cif bfactors)
  scores.model_idx_<i>.npz      chai_lab ranking.rank.get_scores():
                                aggregate_score, ptm, iptm, per_chain_ptm,
                                per_chain_pair_iptm, has_inter_chain_clashes,
                                chain_chain_clashes
  confidence.model_idx_<i>.npz  run_chai.persist_confidence_arrays():
                                StructureCandidates.pae[i] / .pde[i] /
                                .plddt[i] saved unchanged. Absent for runs
                                made before that patch.

Candidates are ordered by Chai's own aggregate_score, highest first -- the
same ordering StructureCandidates.sorted() applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CIF_RE = re.compile(r"^pred\.model_idx_(\d+)\.cif$")


class ResultError(RuntimeError):
    pass


@dataclass
class Candidate:
    candidate_index: int  # Chai's model_idx
    rank: int  # 1 = highest aggregate score
    cif_path: Path
    scores_path: Path
    confidence_path: Path | None
    aggregate_score: float
    ptm: float
    iptm: float
    per_chain_ptm: list[float]
    has_inter_chain_clashes: bool

    def load_confidence(self) -> dict[str, np.ndarray] | None:
        """{'pae': (N, N), 'pde': (N, N), 'plddt': (N,)} or None if not persisted."""
        if self.confidence_path is None:
            return None
        with np.load(self.confidence_path, allow_pickle=False) as data:
            return {k: data[k] for k in ("pae", "pde", "plddt") if k in data.files}


def _scalar(value: np.ndarray) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ResultError(f"expected a single score value, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def load_candidates(output_dir: str | Path) -> list[Candidate]:
    output_dir = Path(output_dir)
    candidates = []
    for cif_path in sorted(output_dir.rglob("pred.model_idx_*.cif")):
        m = _CIF_RE.match(cif_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        scores_path = cif_path.with_name(f"scores.model_idx_{idx}.npz")
        if not scores_path.is_file():
            raise ResultError(f"missing scores file for candidate {idx}")
        confidence_path = cif_path.with_name(f"confidence.model_idx_{idx}.npz")
        with np.load(scores_path, allow_pickle=False) as s:
            missing = {"aggregate_score", "ptm", "iptm"} - set(s.files)
            if missing:
                raise ResultError(f"scores file for candidate {idx} lacks {sorted(missing)}")
            candidates.append(
                Candidate(
                    candidate_index=idx,
                    rank=0,
                    cif_path=cif_path,
                    scores_path=scores_path,
                    confidence_path=confidence_path if confidence_path.is_file() else None,
                    aggregate_score=_scalar(s["aggregate_score"]),
                    ptm=_scalar(s["ptm"]),
                    iptm=_scalar(s["iptm"]),
                    per_chain_ptm=(
                        np.asarray(s["per_chain_ptm"]).reshape(-1).astype(float).tolist()
                        if "per_chain_ptm" in s.files else []
                    ),
                    has_inter_chain_clashes=(
                        bool(np.asarray(s["has_inter_chain_clashes"]).reshape(-1).any())
                        if "has_inter_chain_clashes" in s.files else False
                    ),
                )
            )
    candidates.sort(key=lambda c: c.aggregate_score, reverse=True)
    for rank, c in enumerate(candidates, start=1):
        c.rank = rank
    return candidates


def output_files(output_dir: str | Path) -> list[Path]:
    """Every file in a run's output directory, for the results ZIP."""
    output_dir = Path(output_dir)
    return sorted(p for p in output_dir.rglob("*") if p.is_file())
