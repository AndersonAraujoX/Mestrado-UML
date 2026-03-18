import pytest
from src.core.thermodynamics import ThermodynamicsCalculator
import qutip as qt

def test_compute_average_work():
    rho_0 = qt.fock_dm(2, 0) # |0><0|
    rho_f = qt.fock_dm(2, 1) # |1><1|
    
    H = qt.sigmaz() # Energy cost for flipping using sigma_z
    
    work = ThermodynamicsCalculator.compute_average_work(rho_0, rho_f, H, H)
    # sigmaz |0> = 1
    # sigmaz |1> = -1
    # work = <-1> - <1> = -2
    assert pytest.approx(work) == -2.0
