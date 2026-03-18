import numpy as np
from typing import Tuple, Sequence
import qutip

from single_qubit_operators import single_qubit_Hamiltonian, single_qubit_thermal_state
from many_qubit_operators import many_body_hamiltonian_from_local_operators, create_correlated_terms_01
from partial_swap_multi_qubit import partial_SWAP_two_qubits
from qhe_cycle_qtd_quantities import compute_single_qubit_heating, compute_average_work, compute_partition_heating

def coupling_indexes_rainbow_geometry(N):
    return [[j,N-j-1] for j in range(N//2)]

def coupling_indexes_AiBi_geometry(N):
    return [[j,j+N//2] for j in range(N//2)]

def coupling_indexes_nearest_geometry(N):
    return [[j,j+1] for j in range(N//2)]

def create_swap_list_center_out(N):
    half = N//2
    swap_list = [[half-1,half]]
    b = [[[N-j-2, N-j-1], [j, j+1]] for j in range(half, N-1, 1)]
    b = [pair for sublist in b for pair in sublist]
    swap_list.extend(b)
    return swap_list


def multi_qubit_swap_engine_cycle(
    e_A: float, e_B: float, beta_A: float, beta_B: float, 
    alpha_percentage: float, lamb: float, coupling_indices_correlation: Sequence[Tuple[int, int]],
    coupling_indices_swap: Sequence[Tuple[int, int]],
) -> Tuple[float, float, float]:
    """
    Simulates a two-qubit quantum engine cycle.

    The cycle involves thermalization, correlation creation, and work extraction 
    through an evolution governed by a nearest-neighbor swap Hamiltonian.

    Inputs:
        - e_A: Energy gap of qubits on the left A.
        - e_B: Energy gap of qubits on the right
        - beta_A: Inverse temperature (1/kT) of bath coupled to partition A.
        - beta_B: Inverse temperature (1/kT) of bath coupled to partition B.
        - alpha_percentage: Correlation strength as a fraction of partition function.
        - lamb: Coupling strength for the swap Hamiltonian.
        - coupling_indices_correlation: List of index pairs indicating coupled qubits for the correlation stage.
        - coupling_indices_swap: List of index pairs indicating coupled qubits for the swap stage.
    Outputs:
        - W: Work extracted in the cycle.
        - Q: Heat exchanged by all qubits.
    """
    # hamiltonian
    H_A_j = single_qubit_Hamiltonian(e_A)
    H_B_j = single_qubit_Hamiltonian(e_B)
    H_AB = many_body_hamiltonian_from_local_operators(N, [H_A_j, H_B_j] * half)
    
    ### Thermalization - partition A with TA and partition B with TB  
    ZA_j, rho_A_TA_j = single_qubit_thermal_state(H_A_j, beta_A)
    ZB_j, rho_B_TB_j,  = single_qubit_thermal_state(H_B_j, beta_B)

    print('rhoA', rho_A_TA_j)
    print('rhoB', rho_B_TB_j)
    ### Correlation between pairs
    alpha = alpha_percentage / ZA_j / ZB_j

    # creates a dictionary of indexes in such a way that the ordering of the chosen correlation
    # geometry has the pairs as first neighbors
    dict_couplings = {}
    for i, ic in enumerate(np.array(coupling_indices_correlation).flatten()):
        dict_couplings[ic] = i

    rho_AB_0_corr_pair = qutip.tensor(rho_A_TA_j, rho_B_TB_j) 
    rho_AB_0_corr_pair += create_correlated_terms_01(2, [[0,1]], alpha * np.ones(2))

    # tensor correlated pairs
    rho_AB_0 = qutip.tensor(half * [rho_AB_0_corr_pair])

    rho_AB_tau = rho_AB_0.copy()
    ### Work extraction
    
    print('----------------')
    for k, pair in enumerate(coupling_indices_swap):
        #print('***')
        #print(pair)
        # operations should retrieve the indices
        evolution_operator = partial_SWAP_two_qubits(N, dict_couplings[pair[0]], dict_couplings[pair[1]], lamb)

        # applying all swaps between qubits
        rho_AB_tau = evolution_operator * rho_AB_tau * evolution_operator.dag()


    for j in range(N):
        print('rdm final')
        print(rho_AB_tau.ptrace(j))    

    if(N>2):
        for k, pair in enumerate(coupling_indices_correlation):
            pair2 = [dict_couplings[pair[0]], dict_couplings[pair[1]]]
            print('rdm initial pairs', pair, pair2)

            print(rho_AB_0.ptrace(pair2))
            print(rho_AB_tau.ptrace(pair2))            

    # compute individual's qubit heat
    Q = compute_single_qubit_heating(N, rho_AB_0, rho_AB_tau, [H_A_j, H_B_j] * half)

    # computing the work   
    W = compute_average_work(rho_AB_0, rho_AB_tau, H_AB, H_AB)        

    # compute each partition's heat
    A_indices = list(range(0,N,2))
    H_A = many_body_hamiltonian_from_local_operators(half, [H_A_j] * half)
    Q_A = compute_partition_heating(N, rho_AB_0, rho_AB_tau, H_A, A_indices)

    B_indices = list(range(1,N,2))
    H_B = many_body_hamiltonian_from_local_operators(half, [H_B_j] * half)
    Q_B = compute_partition_heating(N, rho_AB_0, rho_AB_tau, H_B, B_indices)

    return W, Q, Q_A, Q_B

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# model parameters
parser.add_argument('-N','--N', type=int, default=4, help='Number of qubits')
parser.add_argument('-H','--beta_H', type=float, default=1, help='Inverse temperature of the hot bath')
parser.add_argument('-C','--beta_C', type=float, default=1, help='Inverse temperature of the cold bath')
parser.add_argument('-A','--e_A', type=float, default=1, help='Level-splitting for half of qubits on the left part of the set')
parser.add_argument('-B','--e_B', type=float, default=2, help='Level-splitting for half of qubits on the right part of the set')
parser.add_argument('-a','--alpha_percentage', type=float, default=1, help='Strenght of entanglement parameter alpha. alpha_percentage = 1 means \alpha = 1 / Z_A Z_B ')
parser.add_argument('-J','--J', type=float, default=1.0, help='Strenght of the spin-spin interactions.')
parser.add_argument('-L','--lamb', type=float, default=1.0, help='Total time for the duration of the work extraction stage. Lambda equivalent in the original paper. Must be between 0 and 1. If 1, that means that a complete swap has taken place.')
parser.add_argument('-S','--save_results', type=bool, default=False, help='Boolean indicating if the results should be saved or not.')


opts = parser.parse_args()

N = opts.N
beta_A = opts.beta_H
beta_B = opts.beta_C
e_A = opts.e_A
e_B = opts.e_B
alpha_percentage = opts.alpha_percentage
J = opts.J
lamb = opts.lamb
half = N//2
save_res = opts.save_results

# variables for countor plots
nbetaBs = 21
beta_Bs = np.linspace(0,4.0, nbetaBs)

neBs = 21
e_Bs = np.linspace(0,2.0, neBs)


Ws_uncorr = np.zeros((neBs,nbetaBs))
Qjs_uncorr = np.zeros((neBs,nbetaBs, N))
QAs_uncorr = np.zeros((neBs,nbetaBs))
QBs_uncorr = np.zeros((neBs,nbetaBs))

Ws_corr = np.zeros((neBs,nbetaBs))
Qjs_corr = np.zeros((neBs,nbetaBs, N))
QAs_corr = np.zeros((neBs,nbetaBs))
QBs_corr = np.zeros((neBs,nbetaBs))

# correlation couplings - rainbow
coupling_indices_correlation = coupling_indexes_rainbow_geometry(N)

# swap couplings - swap from center to the boundaries and stops at half
coupling_indices_swap = coupling_indexes_nearest_geometry(N)
lc = len(coupling_indices_swap)
coupling_indices_swap = coupling_indices_swap[0:lc//2+1]
#coupling_indexes_AiBi_geometry(N)#create_swap_list_center_out(N)

iB = 10

for eb, e_B in enumerate(e_Bs):
    for bb, beta_B in enumerate(beta_Bs):
        print(eb, bb)
        Ws_uncorr[eb,bb], Qjs_uncorr[eb,bb,:], QAs_uncorr[eb,bb], QBs_uncorr[eb,bb] = multi_qubit_swap_engine_cycle(e_A, e_B, beta_A, beta_B, 0.0, lamb, coupling_indices_correlation, coupling_indices_swap)
        Ws_corr[eb,bb], Qjs_corr[eb,bb,:], QAs_corr[eb,bb], QBs_corr[eb,bb] = multi_qubit_swap_engine_cycle(e_A, e_B, beta_A, beta_B, alpha_percentage, lamb, coupling_indices_correlation, coupling_indices_swap)




file_output_name = f'multi_qubit_swap_engine_results_N={N:d}_alpha={alpha_percentage:.2f}_lamb={lamb:.2f}_nbetas={nbetaBs:d}_nes={neBs:d}'

if(save_res):
    np.savetxt(f'{file_output_name}_work_uncorr.txt', Ws_uncorr)
    np.savetxt(f'{file_output_name}_work_corr.txt', Ws_corr)

    for j in range(N):
        np.savetxt(f'{file_output_name}_heat_j={j:d}_uncorr.txt', Qjs_uncorr[:,:,j])
        np.savetxt(f'{file_output_name}_heatj={j:d}_corr.txt', Qjs_corr[:,:,j])    
