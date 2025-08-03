import pytest
from src.models.synaptic_posner import SynapticPosnerModel

def test_calcium_spike():
    model = SynapticPosnerModel()
    peak = model.calcium_spike(0.001)  # At 1ms
    assert peak > model.params.ca_baseline
    assert peak < 1e-3  # Should be in mM range

def test_phosphate_speciation():
    model = SynapticPosnerModel()
    species = model.phosphate_speciation(pH=7.2)
    assert abs(sum(species.values()) - 1.0) < 1e-6  # Should sum to 1