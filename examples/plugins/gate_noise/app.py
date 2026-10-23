"""
A Bell pair on every rank, to see two-qubit gate noise on its own.

Every rank prepares ``(|00⟩ + |11⟩)/√2`` with a Hadamard and a CNOT and
measures both qubits. Without noise the two bits always agree; an outcome in
which they differ is the trace of noise. With the config of this directory
the CNOT is twenty times noisier than the Hadamard, so nearly all of that
disagreement comes from the two-qubit gate.

On the NetQASM backend only the last rank receives its counts inside
``main``, so that is the rank that reports them.

Run from this directory, so that the plugin can be imported::

    netqmpi -n 2 app.py --netqasm --config config.yaml
"""
from netqmpi.sdk.environment import Environment


def main(env: Environment = None):
    comm = env.comm

    with comm:
        circuit = env.create_circuit(num_qubits=2, num_clbits=2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure(0, 0)
        circuit.measure(1, 1)

    if comm.results:
        counts = comm.results
        shots = sum(counts.values())
        disagree = sum(n for key, n in counts.items() if key[0] != key[1])
        print(f"rank {comm.rank}: {counts}")
        print(f"  the two bits disagree in {disagree / shots:.1%} of {shots} shots")
