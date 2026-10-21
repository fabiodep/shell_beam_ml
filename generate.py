import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import h5py
import jax.numpy as jnp
import numpy as np

from pathlib import Path

from utils import load_checkpoint,loading_inputs
from model import states_trajectory
from config import load_config

model_path = Path("models/beam_implicit_euler_direction_all_input_yield")

checkpoint = load_checkpoint(model_path / "model.pkl")
config = load_config(model_path / "config.yaml")
train_state = jax.device_put(checkpoint["train_state"])

with h5py.File("plot_files/dataset_0.0062_69.0.h5", "r") as data_file:
    data = data_file["states"]
    N = np.asarray(data["N"])
    M = np.asarray(data["M"])
    sim_data = data_file["simulation_config"]
    magnitude = float(sim_data["magnitude"][()])
    angle = float(sim_data["angle"][()])
    t = np.asarray(data["t"])

targets = jnp.vstack([N, M])
inputs = loading_inputs(config, jnp.array([magnitude, angle]), len(t))
predictions  = states_trajectory(config, train_state.params.R_matrix, train_state.params.yield_func, inputs, train_state.norm)
predictions = predictions[:, config.latent_dim:2*config.latent_dim]

with open("plot_files/plot_data.csv", "w") as f:
    f.write("t,eps_bar,kappa_bar,N_pred,M_pred,N_target,M_target\n")
    for i in range(len(t)-1):
        f.write(f"{t[i]},{inputs[i,0]},{inputs[i,1]},{predictions[i,0]},{predictions[i,1]},{N[i+1]},{M[i+1]}\n")