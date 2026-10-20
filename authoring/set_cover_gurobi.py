"""Gurobi backend for the LightBender set-cover placement strategy.

The import of :mod:`gurobipy` is deliberately deferred until solve time so the
other placement policies, and the built-in branch-and-bound set-cover solver,
continue to work without a Gurobi installation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


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
    """Validate candidates and remove empty/exactly duplicated coverage sets.

    When two candidates cover exactly the same chunks, the shorter type always
    dominates in the tertiary objective. Keeping only that candidate reduces
    the MIP without changing its lexicographic optimum.
    """
    if num_chunks < 0:
        raise ValueError("num_chunks must be non-negative")
    if len(coverage) != len(lengths):
        raise ValueError("coverage and lengths must contain the same number of candidates")

    valid_chunks = set(range(num_chunks))
    best_for_coverage: Dict[frozenset, Tuple[int, float]] = {}
    for original_index, (covered, length) in enumerate(zip(coverage, lengths)):
        covered_set = set(covered)
        invalid = covered_set - valid_chunks
        if invalid:
            raise ValueError(
                f"candidate {original_index} refers to invalid chunks: {sorted(invalid)}"
            )
        if not covered_set:
            continue
        if length < 0:
            raise ValueError(f"candidate {original_index} has a negative length cost")

        key = frozenset(covered_set)
        incumbent = best_for_coverage.get(key)
        if incumbent is None or length < incumbent[1]:
            best_for_coverage[key] = (original_index, float(length))

    prepared = sorted(best_for_coverage.items(), key=lambda item: item[1][0])
    indices = [value[0] for _, value in prepared]
    filtered_coverage = [set(key) for key, _ in prepared]
    filtered_lengths = [value[1] for _, value in prepared]

    covered_union = set().union(*filtered_coverage) if filtered_coverage else set()
    missing = valid_chunks - covered_union
    if missing:
        raise SetCoverInfeasibleError(
            f"set-cover instance is infeasible; uncovered chunks: {sorted(missing)}"
        )

    return indices, filtered_coverage, filtered_lengths


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


def solve_set_cover_gurobi(
    coverage: Sequence[Set[int]],
    lengths: Sequence[float],
    num_chunks: int,
    *,
    time_limit: Optional[float] = None,
    mip_gap: float = 0.0,
    output_flag: bool = False,
) -> GurobiSetCoverResult:
    """Solve set cover with exact count/overlap/length lexicographic priorities.

    The objective order is:

    1. minimize the number of selected LightBenders;
    2. minimize chunk overlap beyond the required first cover;
    3. minimize the sum of candidate maximum-length types.

    Gurobi's native multi-objective priorities are used rather than fixed
    scalar weights, whose ordering can break as an instance grows.
    """
    if mip_gap < 0:
        raise ValueError("mip_gap must be non-negative")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive when supplied")

    original_indices, filtered_coverage, filtered_lengths = _prepare_candidates(
        coverage, lengths, num_chunks
    )

    if num_chunks == 0:
        return GurobiSetCoverResult([], 0, 0, 0, 0, 0, 0.0, 0, 0.0, False, "OPTIMAL", True)

    remaining = set(range(len(filtered_coverage)))
    greedy_selection: List[int] = []
    greedy_covered: Set[int] = set()
    greedy_overlap = 0
    while len(greedy_covered) < num_chunks:
        best = max(
            remaining,
            key=lambda i: (
                len(filtered_coverage[i] - greedy_covered),
                -filtered_lengths[i],
                -i,
            ),
        )
        gain = filtered_coverage[best] - greedy_covered
        if not gain:
            raise SetCoverInfeasibleError("greedy initialization could not cover every chunk")
        greedy_overlap += len(filtered_coverage[best] & greedy_covered)
        greedy_covered.update(filtered_coverage[best])
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
        model = gp.Model("lightbender_set_cover")
    except gp.GurobiError as exc:
        raise GurobiUnavailableError(
            f"Gurobi could not start; verify the installation and license ({exc})"
        ) from exc

    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.MIPGap = mip_gap
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    candidate_ids = range(len(filtered_coverage))
    chunk_ids = range(num_chunks)
    covering_candidates = {chunk_id: [] for chunk_id in chunk_ids}
    for i, covered in enumerate(filtered_coverage):
        for chunk_id in covered:
            covering_candidates[chunk_id].append(i)

    selected = model.addVars(candidate_ids, vtype=GRB.BINARY, name="selected")
    overlap = model.addVars(chunk_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="overlap")
    greedy_selection_set = set(greedy_selection)
    for i in candidate_ids:
        selected[i].Start = 1.0 if i in greedy_selection_set else 0.0

    for chunk_id in chunk_ids:
        coverage_expr = gp.quicksum(selected[i] for i in covering_candidates[chunk_id])
        model.addConstr(coverage_expr >= 1, name=f"cover_{chunk_id}")
        # Coverage makes the RHS non-negative. Equality is both exact and a
        # tighter relaxation than the prototype's one-sided slack constraint.
        model.addConstr(overlap[chunk_id] == coverage_expr - 1, name=f"overlap_{chunk_id}")

    model.ModelSense = GRB.MINIMIZE
    model.setObjectiveN(
        gp.quicksum(selected[i] for i in candidate_ids),
        0,
        priority=3,
        abstol=0.0,
        reltol=0.0,
        name="lightbender_count",
    )
    model.setObjectiveN(
        gp.quicksum(overlap[m] for m in chunk_ids),
        1,
        priority=2,
        abstol=0.0,
        reltol=0.0,
        name="chunk_overlap",
    )
    model.setObjectiveN(
        gp.quicksum(filtered_lengths[i] * selected[i] for i in candidate_ids),
        2,
        priority=1,
        abstol=0.0,
        reltol=0.0,
        name="length_sum",
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

    filtered_selection = [i for i in candidate_ids if selected[i].X > 0.5]
    selected_indices = [original_indices[i] for i in filtered_selection]

    cover_counts = [0] * num_chunks
    for i in filtered_selection:
        for chunk_id in filtered_coverage[i]:
            cover_counts[chunk_id] += 1
    if any(count == 0 for count in cover_counts):
        raise RuntimeError("Gurobi returned a solution that does not cover every chunk")

    overlap_value = sum(count - 1 for count in cover_counts)
    length_sum = sum(filtered_lengths[i] for i in filtered_selection)
    greedy_length_sum = sum(filtered_lengths[i] for i in greedy_selection)
    better_than_greedy = (
        len(filtered_selection), overlap_value, length_sum
    ) < (
        len(greedy_selection), greedy_overlap, greedy_length_sum
    )
    return GurobiSetCoverResult(
        selected_indices=selected_indices,
        candidate_count=len(filtered_coverage),
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
