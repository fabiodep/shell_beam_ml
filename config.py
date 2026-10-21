import jax
import yaml

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Tuple, get_args, get_origin, get_type_hints

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    system: str = "shell" #["beam", "shell"]

    # Utlilities
    save_folder: Optional[str] = None
    data_folder: str = "data"
    beam_dataset: str = "long" #["none", "all", "long", "mixed"]
    shell_data_quantity: int = 250
    eval_interval_yield: int = 1000
    eval_interval_state: int = 10000
    save_interval: int = 100
    safety_eps: float = 1e-8
    seed: int = 0
    prefetch_batches: int = 10
    dtype: str = "float64"

    #Physical params
    h: float = 1.0
    b: float = 1.0
    A: float = b * h
    I: float = b * h**3 / 12
    E: float = 200e3
    nu: float = 0.3
    E_mandel: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] = ((E / (1 - nu**2), nu * E / (1 - nu**2), 0.0), (nu * E / (1 - nu**2), E / (1 - nu**2), 0.0), (0.0, 0.0, E / (1 * (1 + nu))))
    sigma_y: float = 250.0
    norm_state: float = sigma_y / E
    N_y: float = sigma_y * A
    M_y: float = sigma_y * (b * h**2 / 6)

    #Optimizer
    beta_1: float = 0.9
    beta_2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0

    #Scheduler
    lr_decay_steps: int = 10
    lr_decay_rate: float = 1.0
    learning_rate_initial_yield: float = 1e-4
    learning_rate_initial: float = 1e-4
    learning_rate_low: float = 1e-4
    high_and_low_lr: bool = False
    high_steps: int = 500
    yield_params_dumping: bool = False
    yield_params_dumping_factor: float = 1e-3

    #Newton solver
    newton_max_iter: int = 20
    newton_atol: float = 1e-8
    newton_line_search_alpha0: float = 0.9
    newton_line_search_beta: float = 0.5
    newton_line_search_c: float = 1e-4
    newton_line_search_max_iter: int = 10

    #Training settings
    num_epochs_yield: int = 25000
    batch_size_yield: int = 10000
    num_epochs: int = 500000
    batch_size: int = 10000
    batch_size_trajectory: int = 100
    pretraining: bool = False
    separated_training: bool = True

    #Loss setting
    loss: str = "L1" #["L2", "L1"]
    normalize_loss: str = "constant" #["constant", "max", "weight","none"]
    derivative_loss: bool = False
    derivative_loss_weight: float = 1e0
    direction_loss: bool = True
    yield_log_loss: bool = False
    peak_loss: bool = False
    peak_loss_weight: float = 1e-1
    reg_state_func: bool = False
    reg_yield_func: bool = False
    l2_reg_weight: float = 1e-5

    #Model settings
    solver: str = "implicit_euler"
    latent_dim: Optional[int] = None
    Dt: float = 1e-3
    R_matrix_shape: Optional[Tuple[int, ...]] = None
    conservative_term: bool = False
    J_matrix_shape: Optional[Tuple[int, ...]] = None
    yield_func_smooth_shape: Optional[Tuple[int, ...]] = None
    yield_func_pos_shape: Optional[Tuple[int, ...]] = None
    yield_func_strength_shape: Optional[Tuple[int, ...]] = None
    yield_func_states_shape: Optional[Tuple[int, ...]] = None
    input_yield: bool = True
    yield_func_type: str = "single" #["single", "complex", "asymp", "composed"]
    diagonal_R: bool = True
    symmetric: bool = True
    features: bool = False
    features_type: str = "mixed" #["atan", "rs", "mixed"]
    symmetry_abs: bool = False

    def __post_init__(self):
        # None selects the original calculated default; explicit values are kept.
        def set_default(name, value):
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)

        if self.solver not in ("implicit_euler", "explicit_euler"):
            raise ValueError(f"Unknown solver: {self.solver}")
        for name in ("num_epochs", "batch_size", "eval_interval_yield", "eval_interval_state", "save_interval", "lr_decay_steps", "prefetch_batches"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        save_folder = f"models/{self.system}_{self.solver}"
        if self.direction_loss:
            save_folder += "_direction"
        else:
            if self.loss == "L1":
                save_folder += "_L1"
            if self.normalize_loss == "constant":
                save_folder += "_normalized"
            elif self.normalize_loss == "max":
                save_folder += "_normalized_max"
            elif self.normalize_loss == "weight":
                save_folder += "_normalized_weight"
            if self.peak_loss:
                save_folder += "_peak"
            if self.derivative_loss:
                save_folder += "_derivative"
            if self.separated_training:
                save_folder += "_separated"
            if self.pretraining:
                save_folder += "_pretraining"
        if self.beam_dataset == "long" and self.system == "beam":
            save_folder += "_long"
        elif self.beam_dataset == "all" and self.system == "beam":
            save_folder += "_all"
        if self.yield_func_type != "single":
            save_folder += f"_{self.yield_func_type}"
        if self.input_yield:
            save_folder += "_input_yield"
        if not self.diagonal_R:
            save_folder += "_full"
        if not self.symmetric:
            save_folder += "_asym"
        if self.features:
            save_folder += f"_{self.features_type}"
        if self.yield_params_dumping:
            save_folder += f"_dmp{self.yield_params_dumping_factor}"
        if self.system == "shell":
            save_folder += f"_{self.shell_data_quantity}"
        set_default("save_folder", save_folder)
        
        if self.system == "beam":
            latent_dim = 2
        elif self.system == "shell":
            latent_dim = 6
        set_default("latent_dim", latent_dim)
        latent_dim = self.latent_dim

        if self.diagonal_R and self.system == "beam":
            R_matrix_shape = (2 * latent_dim, 64, 64, latent_dim)
        elif self.diagonal_R and self.system == "shell":
            dim = latent_dim // 2
            if self.symmetric: 
                R_matrix_shape = (12, 64, 64, dim * (dim + 1))
            else:
                R_matrix_shape = (2 * latent_dim, 64, 64, dim * (dim + 1))
        else:
            R_matrix_shape = (2 * latent_dim, 64, 64, latent_dim * (latent_dim + 1) // 2)
        set_default("R_matrix_shape", R_matrix_shape)

        J_matrix_shape = (2 * latent_dim, 64, 64, (latent_dim-1) * latent_dim // 2)
        set_default("J_matrix_shape", J_matrix_shape)

        if self.system == "beam":
            if self.features:
                if self.features_type == "mixed":
                    input_yield = 3 * latent_dim
                else:
                    input_yield = 3 * latent_dim
            else:
                input_yield = latent_dim

            if self.yield_func_type != "composed":
                yield_func_states_shape = (input_yield, 32, 32, 32, 1)
                set_default("yield_func_states_shape", yield_func_states_shape)
            else:
                yield_func_smooth_shape = (input_yield, 32, 32, 32, 1)
                yield_func_pos_shape = (input_yield, 32, 32, 32, 1)
                yield_func_strength_shape = (input_yield, 32, 32, 32, 1)
                set_default("yield_func_smooth_shape", yield_func_smooth_shape)
                set_default("yield_func_pos_shape", yield_func_pos_shape)
                set_default("yield_func_strength_shape", yield_func_strength_shape)
        elif self.system == "shell":
            if self.symmetric and self.input_yield:
                input_yield = 5
            elif self.symmetric and not self.input_yield:
                input_yield = 3
            elif not self.symmetric:
                input_yield = self.latent_dim

            yield_func_states_shape = (input_yield, 32, 32, 32, 1)
            set_default("yield_func_states_shape", yield_func_states_shape)
        else:
            raise ValueError(f"Unknown configuration: system={self.system}, symmetric={self.symmetric}")

        # YAML represents even nested tuples as lists (for example E_mandel).
        def nested_tuple(value):
            if isinstance(value, (list, tuple)):
                return tuple(nested_tuple(element) for element in value)
            return value

        # Normalize sequence inputs so the frozen configuration stays hashable.
        type_hints = get_type_hints(type(self))
        for item in fields(self):
            value = getattr(self, item.name)
            hint = type_hints[item.name]
            if value is not None and (
                get_origin(hint) is tuple
                or any(get_origin(arg) is tuple for arg in get_args(hint))
            ):
                object.__setattr__(self, item.name, nested_tuple(value))
        object.__setattr__(self, "save_folder", str(self.save_folder))

        save_path = Path(self.save_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(asdict(self), stream, sort_keys=False)


def load_config(path: str | Path) -> Config:
    """Load all YAML fields and save via Config initialization.

    ``path`` may name the YAML file or a directory containing ``config.yaml``.
    Explicit values are preserved, including paths and model shapes. Omitted or
    null derived fields use the same calculated defaults as a new Config.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "config.yaml"
    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("Configuration YAML must contain a mapping of field names to values")

    config_fields = {item.name: item for item in fields(Config)}
    unknown_fields = values.keys() - config_fields.keys()
    if unknown_fields:
        raise ValueError(f"Unknown configuration fields: {', '.join(map(str, unknown_fields))}")

    return Config(**values)
