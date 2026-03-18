
import numpy as np
import qutip as qt
import sys
import os

# To allow importing our modules properly
sys.path.append(os.path.join(os.path.dirname(__file__), "Jupyter_colab"))
sys.path.append(os.path.join(os.path.dirname(__file__), "auxiliar_git"))

from Single_Process import ergotropy_XX, ergotropy_int_troca

def test_ergotropy_xx_basic():
    N = 4
    # Mock parameters
    params_ther = [0.1, 0.2]
    params_corr = [0.1, 0.2, 0.3, 0.4]
    pair_corr = [[0, 3], [1, 2]]
    pair_work = [[0, 1], [1, 2], [2, 3]]
    
    # H_AB mock (identity)
    H_AB = qt.tensor([qt.qeye(2)] * (N-2))
    
    ergo, params = ergotropy_XX(H_AB, params_ther, params_corr, pair_corr, pair_work, N)
    assert ergo is not None
    assert len(params) == len(pair_work)

def test_ergotropy_int_troca_basic():
    N = 4
    # Mock parameters
    params_ther = [0.1, 0.2]
    params_corr = [0.1, 0.2, 0.3, 0.4]
    pair_corr = [[0, 3], [1, 2]]
    pair_work = [[0, 1], [1, 2], [2, 3]]
    
    # H_AB mock (identity)
    H_AB = qt.tensor([qt.qeye(2)] * (N-2))
    
    ergo, params = ergotropy_int_troca(H_AB, params_ther, params_corr, pair_corr, pair_work, N)
    assert ergo is not None
    assert len(params) == 3 * len(pair_work)

if __name__ == "__main__":
    test_ergotropy_xx_basic()
    test_ergotropy_int_troca_basic()
    print("All tests passed!")
