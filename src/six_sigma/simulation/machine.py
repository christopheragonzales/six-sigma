from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class MachineConfig:
    machine_id: str
    cycle_time_mean: float
    cycle_time_std: float
    units_per_cycle: int = 1


class Machine:
    def __init__(self, config: MachineConfig, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def simulate(self, n_cycles: int) -> pl.DataFrame:
        cycle_times = self.rng.normal(
            loc=self.config.cycle_time_mean,
            scale=self.config.cycle_time_std,
            size=n_cycles,
        )

        return pl.DataFrame(
            {
                "machine_id": [self.config.machine_id] * n_cycles,
                "cycle_id": np.arange(1, n_cycles + 1),
                "cycle_time": cycle_times,
                "units_produced": [self.config.units_per_cycle] * n_cycles,
            }
        )
