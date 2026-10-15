import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from pathlib import Path

from utils import load_checkpoint
from model import state_trajectory_vmap, phi_func_vmap

sigma_y = 250.0
E = 200e3
h = 1.0

eps_p_bar_range = jnp.linspace(0.0, 10 * sigma_y / E, 100)
kappa_bar_range = jnp.linspace(0.0, 10 * sigma_y / (E * h), 100)

X, Y = jnp.meshgrid(eps_p_bar_range, kappa_bar_range)

model_path = Path("models/implicit_euler_L1_normalized_pretraining_long_input_yield_dmp0.001")

checkpoint = load_checkpoint(model_path / "model_checkpoint.pkl")
config = checkpoint["config"]
train_state = jax.device_put(checkpoint["train_state"])

X_flat = X.reshape(-1, 1)
Y_flat = Y.reshape(-1, 1)
states_flat = jnp.hstack([X_flat, Y_flat])

phi_z_flat = phi_func_vmap(config, train_state.params.yield_func, states_flat, states_flat, train_state.norm)

with h5py.File(model_path / "yield_func.h5", "w") as f:
    f.create_dataset("eps", data=np.asarray(X_flat.flatten()))
    f.create_dataset("kappa", data=np.asarray(Y_flat.flatten()))
    f.create_dataset("phi_z", data=np.asarray(phi_z_flat))