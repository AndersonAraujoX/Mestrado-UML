'''

This code simulates a quantum thermodynamical cycle that uses N qubits 
as working substance and uses correlations to improve the work extraction.
The cycle comprises 3 stages:

I. Thermalization: half of the qubits are thermalized with a hot bath (T_A) 
and the other with a cold bath (T_B). For example, a system with 8 qubits 
will be thermalized with the following configuration:

 q_0    q_1    q_2    q_3    q_4    q_5    q_6    q_7
_____  _____  _____  _____  _____  _____  _____  _____
 T_A    T_A    T_A    T_A    T_B    T_B    T_B    T_B

There are some flexibility deciding on the Hamiltonian for each qubit.
We chose
H_j = - \epsilon_j / 2 \sigma_j^z
 
II. Correlation: pairs of qubits at different temperature will be entangled.
The resulting mixed state that describes the pair has the form

\rho_ij = \rho_i^T_A \otimes \rho_j^T_A + \chi_{AB}(\alpha)
where
\chi_{AB}(\alpha) = \alpha|0i 1j><1i 0j| +  \alpha^*|1i 0j><0i 1j|

Different geometries of correlation are possible. For example, for 
8 qubits, possibilities are:
(i) j, j+N/2 
(q_0, q_4), (q_1, q_5), (q_2, q_6), (q_3, q_7)

(ii) j, N-j-1
(q_0, q_7), (q_1, q_6), (q_2, q_5), (q_3, q_4)

This is easy to implement by just adding the following interaction terms

\alpha \sigma^+(q_i) \sigma^-(q_j)  + \alpha^\dagger \sigma^-(q_i) \sigma^+(q_j)

We will compare both approaches.

III. Work extraction: the set of qubits will interact with their first neighbors
with Heisenberg interactions for a time \tau. The Hamiltonian that dictates the 
work protocol is

H = -J \sum_{j=0}^{L-2} \vec{\sigma}_j . \vec{\sigma}_{j+1}

where \vec{\sigma}_j = (\sigma_x, \sigma_y, \sigma_z) is a vector composed of 
Pauli matrices.

'''

import numpy as np
import qutip

from single_qubit_operators import single_qubit_Hamiltonian, single_qubit_thermal_state
from many_qubit_operators import many_body_hamiltonian_from_local_operators, create_correlated_terms_01
from partial_swap_multi_qubit import swap_hamiltonian_nearest_neighbors
from spin_chains_hamiltonians import heisenberg_hamiltonian
from qhe_cycle_qtd_quantities import compute_single_qubit_heating, compute_average_work


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
parser.add_argument('-T','--tau', type=float, default=1.0, help='Total time for the duration of the work extraction stage. Lambda equivalent in the original paper. Must be between 0 and 1. If 1, that means that a complete swap has taken place.')
parser.add_argument('-U','--evolution_swap_or_heisenberg', type=bool, default=0, help='')

opts = parser.parse_args()

N = opts.N
beta_A = opts.beta_H
beta_B = opts.beta_C
e_A = opts.e_A
e_B = opts.e_B
alpha_percentage = opts.alpha_percentage
J = opts.J
tau = opts.tau * np.pi / 2 / J
lamb = opts.tau
half = N//2
USH = opts.evolution_swap_or_heisenberg


H_A_j = single_qubit_Hamiltonian(e_A)
H_B_j = single_qubit_Hamiltonian(e_B)

ZA_j, rho_A_TA_j = single_qubit_thermal_state(H_A_j, beta_A)
ZB_j, rho_B_TB_j,  = single_qubit_thermal_state(H_B_j, beta_B)

alpha = alpha_percentage / ZA_j / ZB_j

# thermalization
rho_AB_0_uncorr = qutip.tensor(half * [rho_A_TA_j] + half * [rho_B_TB_j])

# correlation
coupling_indices_rainbow = [[j,N-j-1] for j in range(half)]
coupling_vals = alpha * np.ones(half)

rho_AB_corr_terms_rainbow = create_correlated_terms_01(N, coupling_indices_rainbow, coupling_vals)
rho_AB_corr_rainbow = rho_AB_0_uncorr + rho_AB_corr_terms_rainbow

# work extraction
evolution_operator = swap_hamiltonian_nearest_neighbors(N,lamb * np.ones(N))
if(USH):
    work_hamiltonian = heisenberg_hamiltonian(N, J/2 *np.ones(N-1))
    evolution_operator = (-1J * work_hamiltonian * tau).expm()

rho_AB_0 = rho_AB_corr_rainbow
rho_AB_tau = evolution_operator * rho_AB_0 * evolution_operator.dag() # alternatively rho_AB_tau_2 = qutip.mesolve(work_hamiltonian, rho_AB_0, np.linspace(0, tau, 101)).states[-1]

# compute individual's qubit heat
Q_js = compute_single_qubit_heating(N, rho_AB_0_uncorr, rho_AB_tau, [H_A_j] * half + [H_B_j] * half)

# computing the work
H_AB = many_body_hamiltonian_from_local_operators(N, [H_A_j] * half + [H_B_j] * half)
average_work = compute_average_work(rho_AB_0, rho_AB_tau, H_AB, H_AB)