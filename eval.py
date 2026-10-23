import jax.numpy as jnp

from jax import vmap

from model import states_trajectory, phi_func_vmap
from utils import loading_inputs, prefetch_to_device

def eval_single_fn(config, params, input_char, targets, norm):
    """ 
    Evaluates the model on a single batch of data.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        input_char: input characteristics for the evaluation.
        targets: target values for the evaluation.
        norm: normalization factor.
    Returns:
        L2_error: L2 error between the predicted and target values.
        rel_L2_error: relative L2 error between the predicted and target values.
        forces: predicted forces from the model.
    """
    inputs = loading_inputs(config, input_char, targets.shape[0]+1)

    trajectory = states_trajectory(config, params.R_matrix, params.yield_func, inputs[1:], norm)
    
    forces = trajectory[:, config.latent_dim:2*config.latent_dim]
    
    L2_error = jnp.linalg.norm(forces - targets, axis=-1)

    L2_error = jnp.mean(L2_error, axis=-1)

    return L2_error, forces

eval_batch_fn = vmap(eval_single_fn, in_axes=(None, None, 0, 0, None))

def eval_state_func_fn(config, params, dataset, norm):
    """ 
    Evaluates the model's state function over a dataset.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        dataset: dataset to evaluate on.
        norm: normalization factor.
    Returns:
        L2_errors: L2 errors for each batch in the dataset.
        rel_L2_errors: relative L2 errors for each batch in the dataset.
        forces: predicted forces for each batch in the dataset.
        targets: target values for each batch in the dataset.
        input_chars: input characteristics for each batch in the dataset.
    """
    dataset.reset(shuffle=True)
    L2_errors = []
    forces = []
    targets = []
    input_chars = []
    for batch in prefetch_to_device(dataset, config.prefetch_batches):
        L2_errors_batch, forces_batch = eval_batch_fn(config, params, batch["input_chars"], batch["targets"], norm)
        L2_errors.append(L2_errors_batch)
        forces.append(forces_batch)
        targets.append(batch["targets"])
        input_chars.append(batch["input_chars"])

        break

    return (
        jnp.concatenate(L2_errors),
        jnp.concatenate(forces),
        jnp.concatenate(targets),
        jnp.concatenate(input_chars),
    )

def eval_yield_func_fn(config, params, dataset, norm):
    """ 
    Evaluates the model's yield function over a dataset.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        dataset: dataset to evaluate on.
        norm: normalization factor.
    Returns:
        mse: mean squared error of the yield function predictions.
    """
    dataset.reset(shuffle=False)
    mse = jnp.array(0.0)
    total = jnp.array(0)
    
    for batch in prefetch_to_device(dataset, config.prefetch_batches):
        loss_batch = jnp.mean(jnp.square(phi_func_vmap(config, params.yield_func, batch["states"], batch["inputs"], norm) - batch["targets_phi_z"]))
        mse += loss_batch * batch["states"].shape[0]
        total += batch["states"].shape[0]

    return mse / total