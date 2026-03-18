import numpy as np
from tqdm import tqdm
from scipy.linalg import expm
from concurrent.futures import ProcessPoolExecutor
# Import the 'concurrent.futures' module
import concurrent.futures
import seaborn as sns
#from scipy.linalg import expm, sinm, cosm
#from qutip_qip.operations import *
#from qutip import Qobj, sigmax, sigmay, sigmaz , tensor, qeye,ptrace
from IPython.display import Image
from tqdm import tqdm
#import qiskit
#from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2,FakeAlgiers,FakeBelemV2
#from qiskit_aer import noise
#from qiskit_aer import AerSimulator
import pandas as pd
import random as random
from qutip_qip.operations import cnot

from typing import Tuple, Sequence

from typing import Sequence
from single_qubit_operators import si, sx, sy, sz, sp, sm, one_proj, zero_proj
from many_qubit_operators import pair_many_qubit_operator

from datetime import datetime
import numpy as np

import math

import qutip
from tqdm import tqdm
import itertools
from single_qubit_operators import single_qubit_Hamiltonian, single_qubit_thermal_state
from many_qubit_operators import many_body_hamiltonian_from_local_operators, create_correlated_terms_01, pair_many_qubit_operator
from partial_swap_multi_qubit import partial_SWAP_two_qubits
from qhe_cycle_qtd_quantities import compute_single_qubit_heating, compute_average_work, compute_partition_heating
import sys
sys.path.append('Jupyter_colab')  # To allow importing src
sys.path.append('auxiliar_git')   # To allow importing single_qubit_operators etc

from src.core.quantum_system import QuantumThermalMachine
from src.core.operations import ThermalizationOperation, ImaginaryParametricCorrelation, TrotterizedHeisenbergXYZ, TrotterizedHeisenbergXX
from src.optimization.ergotropy_optimizer import ErgotropyOptimizer
import qutip as qt

def ergotropy_int_troca(H_AB,params_ther,params_corr,pair_corr,pair_work, N):
  machine = QuantumThermalMachine(num_qubits=N)
  thermal_op = ThermalizationOperation(params_ther[0], params_ther[1])
  corr_op = ImaginaryParametricCorrelation(pair_corr, params_corr)
  
  initial_state_vec = machine.get_initial_state(thermal_op, corr_op)
  rho = qt.Qobj(initial_state_vec)

  optimizer = ErgotropyOptimizer(
      rho0=rho, H_i=H_AB, H_f=H_AB, num_qubits=N, 
      learning_rate=0.005, num_epochs=10000, tol=0.0000001
  )

  initial_params = np.random.random(3*len(pair_work))
  params, loss_vec, ergo = optimizer.optimize(TrotterizedHeisenbergXYZ, pair_work, initial_params)

  return ergo, params

def ergotropy_XX(H_AB,params_ther,params_corr,pair_corr,pair_work, N):
  machine = QuantumThermalMachine(num_qubits=N)
  thermal_op = ThermalizationOperation(params_ther[0], params_ther[1])
  corr_op = ImaginaryParametricCorrelation(pair_corr, params_corr)
  
  initial_state_vec = machine.get_initial_state(thermal_op, corr_op)
  rho = qt.Qobj(initial_state_vec)

  optimizer = ErgotropyOptimizer(
      rho0=rho, H_i=H_AB, H_f=H_AB, num_qubits=N, 
      learning_rate=0.005, num_epochs=3000, tol=0.0000001
  )

  initial_params = np.random.random(len(pair_work))
  params, loss_vec, ergo = optimizer.optimize(TrotterizedHeisenbergXX, pair_work, initial_params)

  return ergo, params
def process_single_eb(eb_index, e_B, N, e_A, beta_A, beta_Bs, pair_corr, pair_work):
    """
    Função que encapsula o trabalho para um único valor de e_B.
    Será executada por um processo separado.
    Retorna os resultados Ws_XX e Ws_int para esta fatia de e_B.
    """
    half = N // 2
    vec_eb = [e_A] * half + [e_B] * half

    vec_H_AB = [0 for _ in range(N)]

    for i in range(half):
        H_A_j = single_qubit_Hamiltonian(vec_eb[i])
        H_B_j = single_qubit_Hamiltonian(vec_eb[i + half])

        vec_H_AB[i] = (H_A_j)
        vec_H_AB[i + half] = (H_B_j)

    vec_H_AB = vec_H_AB[1:-1]

    H_AB = many_body_hamiltonian_from_local_operators(N - 2, vec_H_AB)

    results_XX_for_eb = np.zeros(len(beta_Bs))
    results_int_for_eb = np.zeros(len(beta_Bs))

    # 1. Crie um array para armazenar os parâmetros (o tamanho é len(pair_work))
    num_params_xx = len(pair_work)
    results_params_XX_for_eb = np.zeros((len(beta_Bs), num_params_xx))
    num_params_int = 3*len(pair_work)
    results_params_int_for_eb = np.zeros((len(beta_Bs), num_params_int))

    for bb, beta_B in enumerate(beta_Bs):

        #definições
        b_B = beta_B
        b_A = beta_A

        #parametros de correlação
        sz=np.array([[1, 0], [0, -1]])

        H_A=-0.5*e_A*sz
        H_B=-0.5*e_B*sz

        # Função partição
        Za = np.trace(expm(-b_A * H_A))

        Zb = np.trace(expm(-b_B * H_B))

        y = 0
        #operador densidade

        p_a=np.exp(-e_A*b_A/2)/Za
        p_b=np.exp(-e_B*b_B/2)/Zb

        #probabilidade plus and minus
        pm = p_a * (2 * p_b - 1) - p_b + 1

        denominator_pp = (2 * y - 1) * (-2 * p_a * p_b + p_a + p_b - 1)

        pp= 0.5 * (1 - (-p_a - p_b + 1) / denominator_pp)

        #
        nume_x = y * (p_a - p_b) * (p_a * (2 * p_b - 1) - p_b + 1) + (2 * p_a - 1) * (p_b - 1) * p_b
        deno_x = ( p_a + p_b - 1) * (p_a * (2 * p_b - 1) - p_b)
        x = nume_x/deno_x

        #parametros termicos
        theta_A,theta_B = 2 * np.arccos(np.sqrt(pp)),2 * np.arccos(np.sqrt(pm))
        #coeficiente de correlação


        theta = np.arccos(np.sqrt(y))
        phi = np.arccos(np.sqrt(x))

        ws_xx,params_xx = ergotropy_XX(H_AB, [theta_A, theta_B], [theta, phi] * len(pair_corr), pair_corr, pair_work, N)
        ws_int,params_int = ergotropy_int_troca(H_AB, [theta_A, theta_B], [theta, phi] * len(pair_corr), pair_corr, pair_work, N)

        results_XX_for_eb[bb] = ws_xx
        results_int_for_eb[bb] = ws_int

        results_params_int_for_eb[bb,:] = params_int
        results_params_XX_for_eb[bb,:] = params_xx

    return eb_index, results_XX_for_eb, results_int_for_eb,results_params_XX_for_eb, results_params_int_for_eb

if __name__ == "__main__":
    beta_A = 1
    #beta_B = 2
    e_A = 1
    # variables for countor plots
    beta_Bs = [1.1,1.25,1.5,2]
    nbetaBs = len(beta_Bs)

    neBs = 21
    e_Bs = np.linspace(0.1,1.5, neBs)

    Ns = [4]

    N_s = len(Ns)

    Ws_XX = np.zeros((N_s,neBs,nbetaBs))

    Ws_int = np.zeros((N_s,neBs,nbetaBs))

    # Loop principal para os valores de e_A (assumindo que ns seja o índice de e_A)
    for ns, N in enumerate(tqdm(Ns, desc="Processando e_As")):

        #contador = 0

        # gerando combinações de configurações
        from utils_networks import rainbow_connection, linear_connection_bath # Assumed or missing imports
        set1 = set(tuple(item) for item in rainbow_connection(N)[1:])
        set2 = set(tuple(item) for item in linear_connection_bath(N))
        pair = set1.union(set2)
        pair2 = []
        for i in pair:
          pair2.append(list(i))

        pair_work = pair2

        pair_corr = rainbow_connection(N)
        pair_corr = pair_corr[1:] # retirando os banhos

        # Redefine o array de parâmetros a cada iteração de N (se N mudar)
        num_params = len(pair_work)
        Params_XX_N = np.zeros((neBs, nbetaBs, num_params))
        Params_int_N = np.zeros((neBs, nbetaBs, 3*num_params))

        with ProcessPoolExecutor() as executor:
            # Mapeia a função `process_single_eb` para cada item em `e_Bs`
            # `tqdm` é usado para mostrar o progresso do ProcessPoolExecutor
            futures = [executor.submit(process_single_eb, eb, e_B, N, e_A, beta_A, beta_Bs, pair_corr, pair_work)
                       for eb, e_B in enumerate(e_Bs)]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(e_Bs), desc=f"Paralelizando e_Bs para e_A={e_A:.2f}"):
                eb_index, results_XX_for_eb, results_int_for_eb, results_params_XX_for_eb,results_params_int_for_eb = future.result()
                Ws_XX[ns, eb_index, :] = results_XX_for_eb
                Ws_int[ns, eb_index, :] = results_int_for_eb
                Params_XX_N[eb_index,:,:] = results_params_XX_for_eb
                Params_int_N[eb_index,:,:] = results_params_int_for_eb

    print("Cálculos concluídos!")

