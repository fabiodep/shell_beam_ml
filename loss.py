import jax
import jax.numpy as jnp

from jax import vmap
from jax.tree_util import tree_leaves

from model import states_trajectory, phi_func_vmap, state_dot_implicit_vmap
from utils import loading_inputs

def loss_trajectory(config, params_R_matrix, params_yield_func, input_char, targets, states, norm):
    """ 
    Computes the loss over a trajectory of states given the model parameters and inputs.
    Args:
        config: configuration object.
        params_R_matrix: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the R matrix.
        params_yield_func: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the yield function.
        input_char: input characteristics for the trajectory.
        targets: target forces for the trajectory.
        states: sequence of states over time.
        norm: normalization factor.
    Returns:
        The computed loss over the trajectory.
    """
    inputs = loading_inputs(config, input_char, targets.shape[0]+1)
    trajectory = states_trajectory(config, params_R_matrix, params_yield_func, inputs[1:], norm)
    
    forces = trajectory[:,config.latent_dim:2*config.latent_dim]
    error = forces - targets
    if config.normalize_loss == "constant":
        error = error.at[:, :(config.latent_dim//2)].set(error[:, :(config.latent_dim//2)] / config.N_y)
        error = error.at[:, (config.latent_dim//2):].set(error[:, (config.latent_dim//2):] / config.M_y)
    elif config.normalize_loss == "max":
        error = error.at[:, :(config.latent_dim//2)].set(error[:, :(config.latent_dim//2)] / jnp.max(jnp.abs(targets[:, :(config.latent_dim//2)])))
        error = error.at[:, (config.latent_dim//2):].set(error[:, (config.latent_dim//2):] / jnp.max(jnp.abs(targets[:, (config.latent_dim//2):])))
    elif config.normalize_loss == "weight":
        direction_differences = 1 + jnp.linalg.norm(jnp.abs(jnp.abs(jnp.concatenate([jnp.zeros((1, 2)), states[1:] - states[:-1]], axis=0)) - jnp.abs(inputs[1:, config.latent_dim:])), axis=-1)
        max_difference = jnp.max(direction_differences)
        weight = direction_differences / max_difference

    if config.loss == "L1":
        loss = jnp.mean(jnp.linalg.norm(error, axis=-1, ord=1))
        if config.normalize_loss == "weight":
            loss = jnp.sum(jnp.linalg.norm(error, axis=-1, ord=1) * weight) / jnp.sum(weight)
    else:
        loss = jnp.mean(jnp.linalg.norm(error, axis=-1))
        if config.normalize_loss == "weight":
            loss = jnp.sum(jnp.linalg.norm(error, axis=-1) * weight) / jnp.sum(weight)

    if config.derivative_loss:
        target_directions = states[1:] - states[:-1]
        predicted_directions = trajectory[1:, -1][:, None] * trajectory[1:, 2*config.latent_dim:-1]
        zero_states_pos = jnp.isclose(jnp.linalg.norm(states, axis=-1), 0.0)
        target_directions = jnp.where(zero_states_pos[:-1, None], jnp.zeros_like(target_directions), target_directions)
        predicted_directions = jnp.where(zero_states_pos[:-1, None], jnp.zeros_like(predicted_directions), predicted_directions)
        der_loss = jnp.sum(jnp.linalg.norm(predicted_directions - target_directions, axis=-1)) / jnp.sum(jnp.where(~zero_states_pos, 1, 0))

        loss = loss + config.derivative_loss_weight * der_loss

    if config.peak_loss:
        loss = loss + config.peak_loss_weight * (jnp.max(jnp.abs(error[:, :(config.latent_dim//2)])) + jnp.max(jnp.abs(error[:, (config.latent_dim//2):])))

    return loss

loss_trajectory_vmap = vmap(loss_trajectory, in_axes=(None, None, None, 0, 0, 0, None))

def loss_state_func_fn(config, params_R_matrix, params_yield_func, angles, targets, states, norm):
    """ 
    Computes the loss for the state function of the states given the model parameters and inputs.
    Args:
        config: configuration object.
        params_R_matrix: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the R matrix.
        params_yield_func: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the yield function.
        angles: input angles for the state function.
        targets: target forces for the state function.
        states: sequence of states over time.
        norm: normalization factor.
    Returns:
        The computed loss for the state function.
    """
    losses = loss_trajectory_vmap(config, params_R_matrix, params_yield_func, angles, targets, states, norm)

    if config.reg_state_func:
        l2_reg = sum(jnp.sum(param ** 2) for param in tree_leaves(params_yield_func))
    else:
        l2_reg = 0.0

    if config.asymtotic_yield_func:
        return jnp.mean(losses) + config.l2_reg_weight * l2_reg + jnp.square(params_yield_func.bias)
    else:
        return jnp.mean(losses) + config.l2_reg_weight * l2_reg

def loss_yield_func_fn(config, params_yield_func, states, inputs, targets_phi_z, norm):
    """
    Computes the loss for the yield function based on the predicted and target yield values.
    Args:
        config: configuration object.
        params_yield_func: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the yield function.
        states: sequence of states over time.
        inputs: sequence of inputs over time.
        targets_phi_z: target yield values for the yield function.
        norm: normalization factor.
    Returns:
        The computed loss for the yield function.
    """
    predicted_phi_z = phi_func_vmap(config, params_yield_func, states, inputs, norm)

    if config.reg_yield_func:
        l2_reg = sum(jnp.sum(param ** 2) for param in tree_leaves(params_yield_func))
    else:
        l2_reg = 0.0

    if config.yield_log_loss:
        return jnp.mean(jnp.square(jnp.log(predicted_phi_z) - jnp.log(targets_phi_z))) + config.l2_reg_weight * l2_reg
    else:
        return jnp.mean(jnp.square(predicted_phi_z - targets_phi_z)) + config.l2_reg_weight * l2_reg

def loss_direction(config, params_R_matrix, input_char, states, norm):
    """ 
    Computes the loss for the direction of the states given the model parameters and inputs.
    Args:
        config: configuration object.
        params_R_matrix: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the R matrix.
        input_char: input characteristics for the direction loss.
        states: sequence of states over time.
        norm: normalization factor.
    Returns:
        The computed loss for the direction of the states.
    """
    inputs = loading_inputs(config, input_char, states.shape[0])
    target_directions = (states[1:] - states[:-1]) / config.Dt
    target_directions = target_directions / jnp.linalg.norm(target_directions, axis=-1, keepdims=True)
    predicted_directions = state_dot_implicit_vmap(config, params_R_matrix, states[1:], inputs[1:], norm)
    zero_states_pos = jnp.isclose(jnp.linalg.norm(states, axis=-1), 0.0)
    target_directions = jnp.where(zero_states_pos[:-1, None], jnp.zeros_like(target_directions), target_directions)
    predicted_directions = jnp.where(zero_states_pos[:-1, None], jnp.zeros_like(predicted_directions), predicted_directions)
    return jnp.sum(jnp.linalg.norm(predicted_directions - target_directions, axis=-1)) / jnp.sum(jnp.where(~zero_states_pos, 1, 0))

loss_direction_vmap = vmap(loss_direction, in_axes=(None, None, 0, 0, None))

def loss_direction_state_fn(config, params_R_matrix, input_chars, targets, norm):
    """ 
    Computes the loss for the direction of the states over multiple input characteristics.
    Args:
        config: configuration object.
        params_R_matrix: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the R matrix.
        input_chars: sequence of input characteristics for the direction loss.
        targets: target forces for the direction loss.
        norm: normalization factor.
    Returns:
        The computed loss for the direction of the states over multiple input characteristics.
    """
    losses = loss_direction_vmap(config, params_R_matrix, input_chars, targets, norm)
    return jnp.mean(losses)