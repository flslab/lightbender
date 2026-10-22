"""Gurobi set-cover backend using the model formulation from ``gurobi/f.py``.

This module keeps the reusable interface of :mod:`set_cover_gurobi`, but its
Gurobi model uses the binary variables, overlap inequalities, and weighted
single objective from ``gurobi/f.py``.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple


COUNT_WEIGHT = 1000.0
OVERLAP_WEIGHT = 10.0
LENGTH_WEIGHT = 1.0


class GurobiUnavailableError(RuntimeError):
    """Raised when the Gurobi backend or a usable license is unavailable."""


class SetCoverInfeasibleError(RuntimeError):
    """Raised when at least one required chunk cannot be covered."""


@dataclass(frozen=True)
class GurobiSetCoverResult:
    selected_indices: List[int]
    candidate_count: int
    chunk_count: int
    node_count: int
    greedy_size: int
    greedy_overlap: int
    greedy_length_sum: float
    overlap: int
    length_sum: float
    better_than_greedy: bool
    status: str
    optimal: bool


def _prepare_candidates(
    coverage: Sequence[Set[int]], lengths: Sequence[float], num_chunks: int
) -> Tuple[List[int], List[Set[int]], List[float]]:
    """Validate inputs without changing the candidate set used by the model."""
    if num_chunks < 0:
        raise ValueError("num_chunks must be non-negative")
    if len(coverage) != len(lengths):
        raise ValueError("coverage and lengths must contain the same number of candidates")

    valid_chunks = set(range(num_chunks))
    prepared_coverage: List[Set[int]] = []
    prepared_lengths: List[float] = []
    for candidate_index, (covered, length) in enumerate(zip(coverage, lengths)):
        covered_set = set(covered)
        invalid = covered_set - valid_chunks
        if invalid:
            raise ValueError(
                f"candidate {candidate_index} refers to invalid chunks: {sorted(invalid)}"
            )
        if length < 0:
            raise ValueError(f"candidate {candidate_index} has a negative length cost")
        prepared_coverage.append(covered_set)
        prepared_lengths.append(float(length))

    covered_union = set().union(*prepared_coverage) if prepared_coverage else set()
    missing = valid_chunks - covered_union
    if missing:
        raise SetCoverInfeasibleError(
            f"set-cover instance is infeasible; uncovered chunks: {sorted(missing)}"
        )

    return list(range(len(prepared_coverage))), prepared_coverage, prepared_lengths


def _status_name(grb, status: int) -> str:
    names = {
        grb.OPTIMAL: "OPTIMAL",
        grb.INFEASIBLE: "INFEASIBLE",
        grb.INF_OR_UNBD: "INF_OR_UNBD",
        grb.UNBOUNDED: "UNBOUNDED",
        grb.TIME_LIMIT: "TIME_LIMIT",
        grb.INTERRUPTED: "INTERRUPTED",
        grb.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"STATUS_{status}")


def _weighted_objective(size: int, overlap: int, length_sum: float) -> float:
    return (
        COUNT_WEIGHT * size
        + OVERLAP_WEIGHT * overlap
        + LENGTH_WEIGHT * length_sum
    )


def solve_set_cover_gurobi(
    coverage: Sequence[Set[int]],
    lengths: Sequence[float],
    num_chunks: int,
    *,
    time_limit: Optional[float] = None,
    mip_gap: float = 0.0,
    output_flag: bool = False,
) -> GurobiSetCoverResult:
    """Solve set cover with the weighted binary model from ``gurobi/f.py``.

    The minimized objective is ``1000 * count + 10 * overlap + length``.
    """
    if mip_gap < 0:
        raise ValueError("mip_gap must be non-negative")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive when supplied")

    original_indices, prepared_coverage, prepared_lengths = _prepare_candidates(
        coverage, lengths, num_chunks
    )

    if num_chunks == 0:
        return GurobiSetCoverResult([], len(coverage), 0, 0, 0, 0, 0.0, 0, 0.0, False, "OPTIMAL", True)

    remaining = set(range(len(prepared_coverage)))
    greedy_selection: List[int] = []
    greedy_covered: Set[int] = set()
    greedy_overlap = 0
    while len(greedy_covered) < num_chunks:
        best = max(
            remaining,
            key=lambda i: (
                len(prepared_coverage[i] - greedy_covered),
                -prepared_lengths[i],
                -i,
            ),
        )
        gain = prepared_coverage[best] - greedy_covered
        if not gain:
            raise SetCoverInfeasibleError("greedy initialization could not cover every chunk")
        greedy_overlap += len(prepared_coverage[best] & greedy_covered)
        greedy_covered.update(prepared_coverage[best])
        greedy_selection.append(best)
        remaining.remove(best)

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except (ImportError, ModuleNotFoundError) as exc:
        raise GurobiUnavailableError(
            "the Gurobi set-cover solver requires the optional 'gurobipy' package"
        ) from exc

    try:
        model = gp.Model("LB_cover_LP")
    except gp.GurobiError as exc:
        raise GurobiUnavailableError(
            f"Gurobi could not start; verify the installation and license ({exc})"
        ) from exc

    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.MIPGap = mip_gap
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    candidate_ids = range(len(prepared_coverage))
    chunk_ids = range(num_chunks)
    covering_candidates = {
        chunk_id: [
            candidate_id
            for candidate_id in candidate_ids
            if chunk_id in prepared_coverage[candidate_id]
        ]
        for chunk_id in chunk_ids
    }

    # Match gurobi/f.py: binary selections and nonnegative continuous slacks.
    selected = model.addVars(candidate_ids, vtype=GRB.BINARY, name="x")
    overlap = model.addVars(chunk_ids, lb=0.0, name="o")

    for chunk_id in chunk_ids:
        covering = covering_candidates[chunk_id]
        model.addConstr(
            gp.quicksum(selected[i] for i in covering) >= 1,
            name=f"cover_{chunk_id}",
        )

    for chunk_id in chunk_ids:
        covering = covering_candidates[chunk_id]
        model.addConstr(
            overlap[chunk_id]
            >= gp.quicksum(selected[i] for i in covering) - 1,
            name=f"overlap_{chunk_id}",
        )

    model.setObjective(
        COUNT_WEIGHT * gp.quicksum(selected[i] for i in candidate_ids)
        + OVERLAP_WEIGHT * gp.quicksum(overlap[m] for m in chunk_ids)
        + LENGTH_WEIGHT
        * gp.quicksum(prepared_lengths[i] * selected[i] for i in candidate_ids),
        GRB.MINIMIZE,
    )

    try:
        model.optimize()
    except gp.GurobiError as exc:
        raise GurobiUnavailableError(f"Gurobi failed while solving set cover ({exc})") from exc

    status = _status_name(GRB, model.Status)
    if model.SolCount == 0:
        if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            raise SetCoverInfeasibleError("Gurobi found the set-cover instance infeasible")
        raise RuntimeError(f"Gurobi did not produce a set-cover solution (status: {status})")

    prepared_selection = [i for i in candidate_ids if selected[i].X > 0.5]
    selected_indices = [original_indices[i] for i in prepared_selection]

    cover_counts = [0] * num_chunks
    for candidate_id in prepared_selection:
        for chunk_id in prepared_coverage[candidate_id]:
            cover_counts[chunk_id] += 1
    if any(count == 0 for count in cover_counts):
        raise RuntimeError("Gurobi returned a solution that does not cover every chunk")

    overlap_value = sum(count - 1 for count in cover_counts)
    length_sum = sum(prepared_lengths[i] for i in prepared_selection)
    greedy_length_sum = sum(prepared_lengths[i] for i in greedy_selection)
    better_than_greedy = _weighted_objective(
        len(prepared_selection), overlap_value, length_sum
    ) < _weighted_objective(
        len(greedy_selection), greedy_overlap, greedy_length_sum
    )

    return GurobiSetCoverResult(
        selected_indices=selected_indices,
        candidate_count=len(prepared_coverage),
        chunk_count=num_chunks,
        node_count=int(round(model.NodeCount)),
        greedy_size=len(greedy_selection),
        greedy_overlap=greedy_overlap,
        greedy_length_sum=greedy_length_sum,
        overlap=overlap_value,
        length_sum=length_sum,
        better_than_greedy=better_than_greedy,
        status=status,
        optimal=model.Status == GRB.OPTIMAL,
    )
