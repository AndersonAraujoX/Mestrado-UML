import pytest
import pennylane as qml
import numpy as np

from src.core.operations import (
    ThermalizationOperation, 
    IsotropicCorrelation, 
    ParametricCorrelation,
    TrotterizedHeisenbergXX
)
from src.core.hamiltonians import HeisenbergXX

def test_thermalization_operation():
    op = ThermalizationOperation(0.5, 0.5)
    dev = qml.device('default.qubit', wires=4)

    @qml.qnode(dev)
    def circuit():
        op.apply(4)
        return qml.state()
    
    state = circuit()
    # Ensure it runs without errors
    assert len(state) == 16


def test_parametric_correlation():
    pairs = [[1, 2]]
    params = [0.1, 0.2]  # theta, phi
    op = ParametricCorrelation(pairs, params)
    
    dev = qml.device('default.qubit', wires=4)
    @qml.qnode(dev)
    def circuit():
        op.apply(4)
        return qml.state()

    state = circuit()
    assert len(state) == 16


def test_heisenberg_hamiltonian():
    builder = HeisenbergXX([[0, 1]], [1.0])
    obs = builder.build()
    
    # Check it produced a valid PennyLane observable
    assert isinstance(obs, qml.operation.Operator)


def test_trotterized_xx():
    op = TrotterizedHeisenbergXX([[0,1]], [1.0], time=0.1, n_trotter=1)
    
    dev = qml.device('default.qubit', wires=2)
    @qml.qnode(dev)
    def circuit():
        op.apply(2)
        return qml.state()

    state = circuit()
    assert len(state) == 4
