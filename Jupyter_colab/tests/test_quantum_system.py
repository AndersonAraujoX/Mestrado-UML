import pytest
import qutip as qt
import numpy as np
from src.core.quantum_system import QuantumThermalMachine
from src.core.operations import ThermalizationOperation, ParametricCorrelation, TrotterizedHeisenbergXX

def test_quantum_machine():
    machine = QuantumThermalMachine(num_qubits=2, type_dev="default.qubit")
    
    therm = ThermalizationOperation(np.pi/4, np.pi/4)
    state = machine.get_initial_state(therm, None)
    
    assert state is not None
    assert len(state) == 4
    
    work_op = TrotterizedHeisenbergXX([[0,1]], [1.0])
    qutip_state = qt.Qobj(state)
    
    # Since apply_work_operator applies inside QNode, qutip state goes into StatePrep
    final_state = machine.apply_work_operator(qutip_state, work_op)
    assert len(final_state) == 4
