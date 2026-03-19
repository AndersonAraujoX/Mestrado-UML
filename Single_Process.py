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

import sys
sys.path.append('Jupyter_colab')  # To allow importing src
sys.path.append('auxiliar_git')   # To allow importing single_qubit_operators etc

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

    # Listas para armazenar os resultados de cada repetição
    results_Ei_per_repetition = {bb: [] for bb in range(len(beta_Bs))}
    results_Ef_per_repetition = {bb: [] for bb in range(len(beta_Bs))}

    for _ in range(num_repetitions): # Loop para as repetições
        for bb, beta_B in enumerate(beta_Bs):

            #definições
            b_B = beta_B
            b_A = beta_A

            theta_A, theta_B = 2 * np.arctan(np.exp(e_B * b_B * 0.5)), 2 * np.arctan(np.exp(e_A * b_A * 0.5))

            sz = np.array([[1, 0], [0, -1]])

            H_A = -0.5 * e_A * sz
            H_B = -0.5 * e_B * sz

            Za = np.trace(expm(-b_A * H_A))
            Zb = np.trace(expm(-b_B * H_B))

            p_a = np.exp(-e_A * b_A / 2) / Za
            p_b = np.exp(-e_B * b_B / 2) / Zb

            p00 = p_a * p_b
            p01 = p_a * (1 - p_b)
            p10 = p_b * (1 - p_a)
            p11 = (1 - p_b) * (1 - p_a)

            numero1 = 1

            numero2 = 1

            theta, phi = np.arcsin(numero1) / 2, np.arcsin(numero2) / 2

            # Certifique-se de que params_work tenha as dimensões corretas
            # Otimização feita com 3*len(pair_work) para int_troca, e len(pair_work) para XX
            # Aqui, estamos usando parametros_int, que deve ter 3*len(pair_work)
            current_params_work = list(parametros_work[bb][eb_index])

            w_i, w_f = work_shot(H_AB, [theta_A, theta_B], [theta, phi] * len(pair_corr), pair_corr, current_params_work, pair_work, e_B)

            results_Ei_per_repetition[bb].append(w_i)
            results_Ef_per_repetition[bb].append(w_f)
            
    return eb_index, results_Ei_per_repetition, results_Ef_per_repetition

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
        # Substituído a dependência faltante utils_networks pela lógica local iterativa
        
        # pair_corr é o rainbow connection sem os banhos [1, 2] para N=4
        pair_corr = [[i, N - i - 1] for i in range(1, N // 2)]
        
        # pair_work é a união da parte central com os banhos térmicos (0->1 e (N-2)->(N-1))
        pair_bath = [[0, 1], [N - 2, N - 1]]
        pair_work = pair_bath + pair_corr

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

    import matplotlib.pyplot as plt

    # Configuração da Matplotlib Latex
    font = {'weight': 'bold', 'size': 16}
    # Removido usetex pois o backend as vezes não possui Latex instalado,
    # vamos usar MathText nativo do matplotlib
    plt.rc('font', **font)

    # Plotando os resultados de Ws_XX para ns = 0 (N = 4)
    plt.figure(figsize=(10, 6))
    
    # Eb values na abscissa (pois e_A = 1, então x = e_B / e_A)
    x_axis = e_Bs / e_A
    
    for bb, beta_B in enumerate(beta_Bs):
        # A razão beta_B / beta_A
        beta_ratio = beta_B / beta_A
        # Valores de Ws_XX para todo eb_index com o beta_index específico
        y_values = Ws_XX[0, :, bb] 
        
        plt.plot(
            x_axis, 
            y_values, 
            marker='o', 
            linestyle='-', 
            label=f'$\\beta_B/\\beta_A={beta_ratio:.2f}$'
        )

    plt.xlabel(r'$\epsilon_B / \epsilon_A$')
    plt.ylabel(r'$W_{s}^{XX}$')
    plt.title(r'Best $W_{s}^{XX}$ vs $\epsilon_B / \epsilon_A$ for different $\beta_B / \beta_A$')
    plt.legend()
    plt.grid(True)
    
    # Salvar o gráfico e mostrar (caso em ambiente iterativo)
    plt.savefig('Ws_XX_plot.png', dpi=300, bbox_inches='tight')
    print("Gráfico do Ws_XX salvo como 'Ws_XX_plot.png'")
    # Não chamamos plt.show() em um .py executado em background para não travar o terminal, 
    # mas caso precise descomente:
    # plt.show()

