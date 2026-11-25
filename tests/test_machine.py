import polars as pl

from six_sigma.simulation.machine import Machine, MachineConfig


def test_simulate() -> None:
    config = MachineConfig(
        machine_id="M001", cycle_time_mean=30.0, cycle_time_std=1.5, units_per_cycle=2
    )

    machine = Machine(config, seed=42)
    result = machine.simulate(n_cycles=100)

    assert isinstance(result, pl.DataFrame)
    assert result.height == 100
    assert result["machine_id"].unique().to_list() == ["M001"]
    assert result["units_produced"].sum() == 200


def test_reproducibility() -> None:
    config = MachineConfig(
        machine_id="M001",
        cycle_time_mean=30.0,
        cycle_time_std=1.5,
    )
    machine1 = Machine(config, seed=42).simulate(100)
    machine2 = Machine(config, seed=42).simulate(100)

    assert machine1.equals(machine2)
