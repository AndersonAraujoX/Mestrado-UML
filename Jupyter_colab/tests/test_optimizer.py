import pytest
import numpy as np
import qutip as qt
from src.optimization.ergotropy_optimizer import ErgotropyOptimizer
from src.core.operations import TrotterizedHeisenbergXX

def test_ergotropy_optimizer():
    # Setup simple 4 qubit system (2 internal, 2 bath)
    rho_0 = qt.basis([2, 2, 2, 2], [0, 0, 0, 0])
    H_i = qt.tensor(qt.sigmaz(), qt.sigmaz())
    H_f = qt.tensor(qt.sigmaz(), qt.sigmaz())
    
    optimizer = ErgotropyOptimizer(
        rho0=rho_0,
        H_i=H_i, 
        H_f=H_f,
        num_qubits=4, 
        bath=False,  # Use all qubits for trace context
        num_epochs=1 # Just run 1 epoch to ensure it doesn't crash
    )
    
    # Run optimizer with 1 param
    initial_params = np.array([0.5])
    pair_work = [[0, 1]]
    
    params, loss_vec, final_loss = optimizer.optimize(
        TrotterizedHeisenbergXX, 
        pair_work, 
        initial_params
    )
    
    assert len(params) == 1
    assert len(loss_vec) > 0
    assert isinstance(final_loss, float)
