from dataclasses import dataclass


@dataclass
class Config:
    seq_len: int = 75
    n_splits: int = 5
    n_seeds: int = 1
    n_epochs: int = 20
    batch_size: int = 32
    lr: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.999)
    label_smoothing: float = 0.01
    aux_loss_weight: float = 0.3