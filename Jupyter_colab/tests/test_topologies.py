import pytest
from src.core.topologies import (
    LinearTopology, RingTopology, StarTopology, 
    CompleteTopology, CompleteInternalTopology, 
    RainbowTopology, RingInternalTopology, LinearInternalTopology
)

def test_linear_topology():
    topology = LinearTopology()
    pairs = topology.get_pairs(4)
    assert pairs == [[0, 1], [1, 2], [2, 3]]

def test_ring_topology():
    topology = RingTopology()
    pairs = topology.get_pairs(4)
    assert pairs == [[0, 1], [1, 2], [2, 3], [3, 0]]

def test_star_topology():
    topology = StarTopology()
    pairs = topology.get_pairs(4)
    assert pairs == [[1, 0], [2, 0], [3, 0]]

def test_complete_topology():
    topology = CompleteTopology()
    pairs = topology.get_pairs(4)
    assert pairs == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

def test_rainbow_topology():
    topology = RainbowTopology()
    assert topology.get_pairs(4) == [[0, 3], [1, 2]]
    assert topology.get_pairs(6) == [[0, 5], [1, 4], [2, 3]]
