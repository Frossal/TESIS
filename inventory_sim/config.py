from dataclasses import dataclass

@dataclass
class SimConfig:
    horizon_days: int = 365
    service_min: float = 0.95
    init_reps: int = 12
    max_reps: int = 60
    ci_rel_target: float = 0.03  
    seed_numpy: int = 42
    seed_random: int = 42
