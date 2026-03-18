import numpy as np
from typing import Sequence
import qutip

def compute_single_qubit_heating(N: int, rho_0: qutip.Qobj, rho_f: qutip.Qobj, H_js: Sequence[qutip.Qobj]) -> np.ndarray:
    """
    Computes the heat absorbed by each qubit in an N-qubit system.

    Inputs:
        - N: Number of qubits in the system.
        - rho_0: Initial density matrix of the full system (Qobj).
        - rho_f: Final density matrix of the full system after evolution (Qobj).
        - H_js: List of single-qubit Hamiltonians (each a Qobj).

    Output:
        - Q_j: Numpy array containing the heat Q_j absorbed by each qubit.
    """
    Q_j = np.zeros(N)

    for j in range(N):
        rho_0_j = rho_0.ptrace(j)
        rho_f_j = rho_f.ptrace(j)
        Q_j[j] = -qutip.expect(H_js[j], rho_f_j - rho_0_j)

    return Q_j

def compute_partition_heating(N: int, rho_0: qutip.Qobj, rho_f: qutip.Qobj, H_p: qutip.Qobj, 
                              p_indices: Sequence[int]) -> float:
    """
    Computes the heat transferred to a specific subsystem of a quantum many-body system.

    Parameters:
        N (int): Total number of qubits.
        rho_0 (qutip.Qobj): Initial density matrix of the system.
        rho_f (qutip.Qobj): Final density matrix of the system.
        H_p (qutip.Qobj): Hamiltonian of the partition.
        p_indices (Sequence[int]): Indices of qubits in the partition.

    Returns:
        float: The heat transferred to the partition.
    """
    rho_0_p = rho_0.ptrace(p_indices)
    rho_f_p = rho_f.ptrace(p_indices) 
    Q_p = -qutip.expect(H_p, rho_f_p - rho_0_p)
    return Q_p


def compute_average_work(rho_0: qutip.Qobj, rho_f: qutip.Qobj, H_0: qutip.Qobj, H_f: qutip.Qobj) -> float:
    """
    Computes the average work done on a quantum system during an evolution.

    Work is defined as the change in the expectation value of the Hamiltonian.

    Inputs:
        - rho_0: Initial density matrix of the system (Qobj).
        - rho_f: Final density matrix of the system after evolution (Qobj).
        - H_0: Initial Hamiltonian of the system (Qobj).
        - H_f: Final Hamiltonian of the system (Qobj).

    Output:
        - Work: The average work done on the system.
    """
    return qutip.expect(H_f, rho_f) - qutip.expect(H_0, rho_0)