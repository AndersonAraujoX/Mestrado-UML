import numpy as np
from typing import Tuple
import qutip

from single_qubit_operators import single_qubit_Hamiltonian, single_qubit_thermal_state
from many_qubit_operators import many_body_hamiltonian_from_local_operators, create_correlated_terms_01
from partial_swap_multi_qubit import partial_SWAP_two_qubits
from qhe_cycle_qtd_quantities import compute_single_qubit_heating, compute_average_work


def two_qubit_engine_cycle(
    e_A: float, e_B: float, beta_A: float, beta_B: float, 
    alpha_percentage: float, lamb: float
) -> Tuple[float, float, float]:
    """
    Simulates a two-qubit quantum engine cycle.

    The cycle involves thermalization, correlation creation, and work extraction 
    through an evolution governed by a nearest-neighbor swap Hamiltonian.

    Inputs:
        - e_A: Energy gap of qubit A.
        - e_B: Energy gap of qubit B.
        - beta_A: Inverse temperature (1/kT) of bath coupled to qubit A.
        - beta_B: Inverse temperature (1/kT) of bath coupled to qubit B.
        - alpha_percentage: Correlation strength as a fraction of partition function.
        - lamb: Coupling strength for the swap Hamiltonian.

    Outputs:
        - W: Work extracted in the cycle.
        - Q_A: Heat exchanged by qubit A.
        - Q_B: Heat exchanged by qubit B.
    """
    # hamiltonian
    H_A_j = single_qubit_Hamiltonian(e_A)
    H_B_j = single_qubit_Hamiltonian(e_B)
    H_AB = many_body_hamiltonian_from_local_operators(N, [H_A_j] * half + [H_B_j] * half)
    
    ### Thermalization
    ZA_j, rho_A_TA_j = single_qubit_thermal_state(H_A_j, beta_A)
    ZB_j, rho_B_TB_j,  = single_qubit_thermal_state(H_B_j, beta_B)

    # uncorrelated product state       
    rho_AB_0_uncorr = qutip.tensor(half * [rho_A_TA_j] + half * [rho_B_TB_j])

    ### Correlation
    alpha = alpha_percentage / ZA_j / ZB_j 
    coupling_indices = [[j,N-j-1] for j in range(half)]
    coupling_vals = alpha * np.ones(half)

    rho_AB_corr_terms = create_correlated_terms_01(N, coupling_indices, coupling_vals)
    rho_AB_0 = rho_AB_0_uncorr + rho_AB_corr_terms

    ### Work extraction
    evolution_operator = partial_SWAP_two_qubits(N, 0, 1, lamb)

    # evolving initial states
    rho_AB_tau = evolution_operator * rho_AB_0 * evolution_operator.dag()

    # compute individual's qubit heat
    Q = compute_single_qubit_heating(N, rho_AB_0, rho_AB_tau, [H_A_j] * half + [H_B_j] * half)

    # computing the work   
    W = compute_average_work(rho_AB_0, rho_AB_tau, H_AB, H_AB)        

    return W, Q[0], Q[1]



# marcela's default parameters
N = 2
half = N//2
beta_A = 1
e_A = 1
lamb = 0.6
alpha_percentage = 1.0

# variables for countor plots
nbetaBs = 21
beta_Bs = np.linspace(0,4.0, nbetaBs)

neBs = 21
e_Bs = np.linspace(0,2.0, neBs)


Ws_uncorr = np.zeros((neBs,nbetaBs))
QAs_uncorr = np.zeros((neBs,nbetaBs))
QBs_uncorr = np.zeros((neBs,nbetaBs))

Ws_corr = np.zeros((neBs,nbetaBs))
QAs_corr = np.zeros((neBs,nbetaBs))
QBs_corr = np.zeros((neBs,nbetaBs))


for eb, e_B in enumerate(e_Bs):
    for bb, beta_B in enumerate(beta_Bs):
        Ws_uncorr[eb,bb], QAs_uncorr[eb,bb], QBs_uncorr[eb,bb] = two_qubit_engine_cycle(e_A, e_B, beta_A, beta_B, 0.0, lamb)
        Ws_corr[eb,bb], QAs_corr[eb,bb], QBs_corr[eb,bb] = two_qubit_engine_cycle(e_A, e_B, beta_A, beta_B, alpha_percentage, lamb)