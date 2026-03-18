import numpy as np
from typing import Sequence, Tuple
import qutip

from single_qubit_operators import si, sx, sy, sz, sp, sm, one_proj, zero_proj

def partial_SWAP_two_qubits(N: int, i: int, j: int, x: float) -> qutip.Qobj:
    """
    Constructs a partial SWAP gate acting on qubits i and j in an N-qubit system.

    Parameters:
        N (int): Number of qubits in the system.
        i (int): Index of the first qubit.
        j (int): Index of the second qubit.
        x (float): Partial SWAP parameter (0 ≤ x ≤ 1).

    Returns:
        qutip.Qobj: The partial SWAP operator.
    """
    pSWAP00 = [si] * N
    pSWAP00[i] = zero_proj
    pSWAP00[j] = zero_proj

    pSWAP11 = [si] * N
    pSWAP11[i] = one_proj
    pSWAP11[j] = one_proj

    pSWAP01 = [si] * N
    pSWAP01[i] = zero_proj
    pSWAP01[j] = one_proj

    pSWAP10 = [si] * N
    pSWAP10[i] = one_proj
    pSWAP10[j] = zero_proj

    pSWAPpm = [si] * N
    pSWAPpm[i] = sp
    pSWAPpm[j] = sm

    pSWAPmp = [si] * N
    pSWAPmp[i] = sm
    pSWAPmp[j] = sp

    pSWAP = qutip.tensor(pSWAP00) + qutip.tensor(pSWAP11) \
        + np.sqrt(1-x) * (qutip.tensor(pSWAP01) + qutip.tensor(pSWAP10)) \
        + np.sqrt(x) * qutip.tensor(pSWAPpm) - np.sqrt(x) * qutip.tensor(pSWAPmp)
    
    return pSWAP



def swap_operator_sequence(N: int, rho_0: qutip.Qobj, couplings_indices: Sequence[Tuple[int, int]], 
                           xpairs: Sequence[float]) -> qutip.Qobj:
    """
    Applies a sequence of partial SWAP operations to a quantum state.

    Parameters:
        N (int): Number of qubits.
        rho_0 (qutip.Qobj): Initial density matrix.
        couplings_indices (Sequence[Tuple[int, int]]): List of qubit pairs to apply the SWAP.
        xpairs (Sequence[float]): List of partial SWAP parameters for each pair.

    Returns:
        qutip.Qobj: Final density matrix after applying the swaps.
    """
    rho_f = rho_0.copy()

    for k, pair in enumerate(couplings_indices):
        swap_operator = partial_SWAP_two_qubits(N, pair[0], pair[1], xpairs[k])

        # applying swaps to pairs of qubits iteratively
        rho_f = swap_operator * rho_f * swap_operator.dag()


    return rho_f