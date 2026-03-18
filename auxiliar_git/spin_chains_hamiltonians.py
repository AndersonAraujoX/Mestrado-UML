import numpy as np
from typing import Sequence
import qutip


from single_qubit_operators import si, sx, sy, sz, sp, sm, one_proj, zero_proj
from many_qubit_operators import pair_many_qubit_operator


def heisenberg_hamiltonian(N: int, couplings_values: Sequence[complex]) -> qutip.Qobj:
    """
    Constructs the Heisenberg Hamiltonian for an N-qubit system with nearest-neighbor interactions.

    Inputs:
        - N: Number of qubits in the system.
        - couplings_values: Sequence of coupling constants (real or complex) for each qubit pair.

    Output:
        - Hh: The Heisenberg Hamiltonian as a QuTiP quantum object (Qobj).
    """
    Hh = 0
    for j in range(N-1):
        Hxx = couplings_values[j] * pair_many_qubit_operator(N, j, j+1, sx, sx)
        Hyy = couplings_values[j] * pair_many_qubit_operator(N, j, j+1, sy, sy)
        Hzz = couplings_values[j] * pair_many_qubit_operator(N, j, j+1, sz, sz)
        Hh += Hxx + Hyy + Hzz
    return Hh