"""
Split a run's results into one histogram per rank.

A copy of the three helpers of ``metrics.py`` in the NetQMPI benchmark
(https://github.com/NetQIR/netqmpi-benchmark) that the example tests need.
The benchmark now lives in its own repository, and a test of NetQMPI should
not depend on it.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _clean(key: Any) -> str:
    """
    Return a histogram key as a bare bit string.

    Backends decorate their keys differently — CUNQA separates registers
    with spaces, Qiskit does the same when a circuit has several — and none
    of that carries meaning for an all-zero criterion.

    Args:
        key: Raw histogram key.

    Returns:
        The key with whitespace removed.
    """
    return str(key).replace(" ", "")


def per_rank_histograms(envs: List[Any], backend: str) -> Dict[int, Dict[str, int]]:
    """
    Normalise a run's results into one histogram per rank.

    The four backends hand back three different shapes: CUNQA gives every
    communicator the whole ``{rank: counts}`` picture, Qoala and NetQASM
    give each communicator its own histogram, and Aer gives every
    communicator the *same* histogram over the classical bits of all ranks
    at once.

    Args:
        envs: The per-rank environments of the run, in rank order.
        backend: Backend name.

    Returns:
        A mapping from rank to that rank's histogram. Empty when the run
        produced no results at all.
    """
    if not envs:
        return {}

    if backend == "aer":
        return _split_joint(envs)

    # CUNQA publishes the complete picture, keyed by rank, on every
    # communicator; the last rank to leave its block is the one that has it.
    for env in envs:
        results = env.comm.results
        if results and all(isinstance(key, int) for key in results):
            return {rank: dict(counts) for rank, counts in results.items()}

    return {env.comm.rank: dict(env.comm.results or {}) for env in envs}


def _split_joint(envs: List[Any]) -> Dict[int, Dict[str, int]]:
    """
    Split Aer's single global histogram into one histogram per rank.

    Aer runs every rank inside one ``QuantumCircuit``, so a histogram key
    covers all the ranks' classical bits at once. Qiskit prints them
    most-significant first, so the key is reversed before slicing to index
    it by classical bit number.

    Args:
        envs: The per-rank environments of the run, in rank order.

    Returns:
        A mapping from rank to that rank's marginal histogram.
    """
    joint = envs[0].comm.results or {}
    per_rank: Dict[int, Dict[str, int]] = {env.comm.rank: {} for env in envs}

    widths = {env.comm.rank: sum(c.num_clbits for c in env.comm.circuits)
              for env in envs}

    for key, count in joint.items():
        bits = _clean(key)[::-1]          # bits[i] is classical bit i
        offset = 0
        for env in envs:
            rank = env.comm.rank
            slice_ = bits[offset:offset + widths[rank]]
            per_rank[rank][slice_] = per_rank[rank].get(slice_, 0) + count
            offset += widths[rank]

    return per_rank
