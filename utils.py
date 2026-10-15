import jax
import pickle

import numpy as np
import jax.numpy as jnp
import itertools as it

from jax import jacfwd, random
from jax.lax import while_loop
from functools import partial
from collections import deque
from pathlib import Path

class DictIterator:
    """
    Iterator class for iterating over a dataset represented as a dictionary of arrays. Each array in the dictionary should have the same length along the first dimension.
    Args:
        config: configuration object containing dataset parameters.
        input_chars: array of input characteristics for the dataset.
        targets: array of target values for the dataset.
        states: array of state values for the dataset.
        split: string indicating the dataset split (e.g., "train", "val", "test").
    Attributes:
        seed: random seed for shuffling the dataset.
        split: dataset split type.
        index: current index in the dataset.
        input_chars: array of input characteristics.
        targets: array of target values.
        states: array of state values.
        batch_size: size of each batch for iteration.
        length: total number of samples in the dataset.
        n_batches: total number of batches in the dataset.
    Methods:
        __len__: returns the number of batches in the dataset.
        reset: resets the iterator and optionally shuffles the dataset.
        __iter__: returns the iterator object.
        __next__: returns the next batch of data from the dataset.
    """
    def __init__(self, config, input_chars, targets, states, split="train"):
        self.seed = config.seed
        self.split = split
        self.index = 0
        self.input_chars = input_chars
        self.targets = np.array(targets, copy=True)
        self.targets[..., 1] *= config.h
        self.states = states
        if self.split == "train":
            self.batch_size = config.batch_size
        else:
            self.batch_size = config.batch_size_val
        self.length = self.input_chars.shape[0]
        if self.length == 0:
            raise ValueError(f"Empty {split} dataset")
        self.n_batches = (self.length + self.batch_size - 1) // self.batch_size
        self.rng = np.random.default_rng(config.seed)

    def __len__(self):
        return self.n_batches

    def reset(self, shuffle=True):
        self.index = 0
        if shuffle:
            perm = self.rng.permutation(self.length)
            self.input_chars = self.input_chars[perm]
            self.targets = self.targets[perm]
            self.states = self.states[perm]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        start = self.index
        self.index += self.batch_size
        return {"input_chars": self.input_chars[start:self.index],
                "targets": self.targets[start:self.index],
                "states": self.states[start:self.index]}


class YieldDictIterator:
    """
    Iterator class for iterating over a dataset specifically for yield function data. Each array in the dataset should have the same length along the first dimension.
    Args:
        config: configuration object containing dataset parameters.
        states: array of state values for the dataset.
        inputs: array of input values for the dataset.
        targets_phi_z: array of target yield function values for the dataset.
        split: string indicating the dataset split (e.g., "train", "val", "test").
    Attributes:
        seed: random seed for shuffling the dataset.
        split: dataset split type.
        index: current index in the dataset.
        states: array of state values.
        inputs: array of input values.
        targets_phi_z: array of target yield function values.
        batch_size: size of each batch for iteration.
        length: total number of samples in the dataset.
        n_batches: total number of batches in the dataset.
    Methods:
        __len__: returns the number of batches in the dataset.
        reset: resets the iterator and optionally shuffles the dataset.
        __iter__: returns the iterator object.
        __next__: returns the next batch of data from the dataset.
    """
    def __init__(self, config, states, inputs, targets, split="train"):
        self.seed = config.seed
        self.split = split
        self.index = 0
        args = np.where(np.all(np.logical_not(np.isclose(states, 0.0)), axis=-1))
        self.states = states[args]
        self.inputs = inputs[args]
        self.targets = targets[args]
        self.batch_size = config.batch_size_yield
        self.length = self.states.shape[0]
        self.n_batches = (self.length + self.batch_size - 1) // self.batch_size
        self.rng = np.random.default_rng(config.seed)

    def __len__(self):
        return self.n_batches

    def reset(self, shuffle=True):
        self.index = 0
        if shuffle:
            perm = self.rng.permutation(self.length)
            self.states = self.states[perm]
            self.inputs = self.inputs[perm]
            self.targets = self.targets[perm]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        start = self.index
        self.index += self.batch_size
        return {"states": self.states[start:self.index],
                "inputs": self.inputs[start:self.index],
                "targets_phi_z": self.targets[start:self.index]}


def line_search(z, config, residual_fn, params_R_matrix, params_yield_func, states, inpt, delta, norm):
    """
    Performs a line search to find an appropriate step size for the Newton method.
    Args:
        z: current state vector.
        config: configuration object.
        residual_fn: function to compute the residual.
        params_R_matrix: parameters for the R matrix.
        params_yield_func: parameters for the yield function.
        states: sequence of states over time.
        inpt: sequence of inputs over time.
        delta: proposed update direction for the state vector.
        norm: normalization factor.
    Returns:
        z_new: updated state vector after line search.
    """

    r0 = residual_fn(z, config, params_R_matrix, params_yield_func, states, inpt, norm)
    phi0 = 0.5 * jnp.vdot(r0, r0)

    def body(body_state):
        i, alpha, z_best, accepted = body_state

        z_trial = z + alpha * delta
        r_trial = residual_fn(z_trial, config, params_R_matrix, params_yield_func, states, inpt, norm)
        phi_trial = 0.5 * jnp.vdot(r_trial, r_trial)

        success = phi_trial <= (1.0 - config.newton_line_search_c * alpha) * phi0

        z_best = jnp.where(success, z_trial, z_best)

        alpha = jnp.where(success, alpha, config.newton_line_search_beta * alpha)

        return i + 1, alpha, z_best, accepted | success

    def cond(body_state):
        i, _, _, accepted = body_state
        return (i < config.newton_line_search_max_iter) & (~accepted)

    initial = (jnp.array(0), jnp.asarray(config.newton_line_search_alpha0), z, jnp.array(False))

    _, _, z_new, _ = while_loop(cond, body, initial)
    return z_new

def newton_solve(config, residual_fn, initial_condition, params_R_matrix, params_yield_func, states, inpt, norm):
    """
    Solves a system of equations using the Newton method.
    Args:
        config: configuration object.
        residual_fn: function to compute the residual.
        initial_condition: initial guess for the solution.
        params_R_matrix: parameters for the R matrix.
        params_yield_func: parameters for the yield function.
        states: sequence of states over time.
        inpt: sequence of inputs over time.
        norm: normalization factor.
    Returns:
        z: solution to the system of equations.
    """

    def residual(z):
        return residual_fn(z, config, params_R_matrix, params_yield_func, states, inpt, norm)

    def body(body_state):
        k, z, err = body_state

        r = residual(z)
        J = jacfwd(residual)(z)
        delta = jnp.linalg.solve(J, -r)

        z_new = line_search(z, config, residual_fn, params_R_matrix, params_yield_func, z[:-1], inpt, delta, norm)
        #z_new = z + config.newton_line_search_alpha0 * delta
        err_new = jnp.linalg.norm(residual(z_new))

        return k+1, z_new, err_new

    def cond(body_state):
        k, z, err = body_state
        
        return jnp.logical_and(
            k < config.newton_max_iter,
            err > config.newton_atol
        )

    k, z, _ = while_loop(cond, body, (jnp.array(0), initial_condition,
                                  jnp.linalg.norm(residual(initial_condition))))
    #jax.debug.print("Newton solver converged in {} iterations with residual norm {}", k, jnp.linalg.norm(residual(z)))

    return z


@jax.tree_util.register_pytree_with_keys_class
class dot_dict(dict):
    """
    A dictionary that allows attribute-style access to its items.
    This class extends the built-in dictionary to allow accessing items using dot notation (e.g., obj.key) in addition to the standard dictionary access (e.g., obj['key']).
    Attributes:
        Inherits all attributes from the built-in dictionary class.
    Methods:
        __setattr__: Sets an attribute in the dictionary.
        __getattr__: Gets an attribute from the dictionary.
        tree_flatten_with_keys: Flattens the dictionary into a tuple of keys and values for JAX tree utilities.
        tree_unflatten: Reconstructs the dictionary from a tuple of keys and values for JAX tree utilities.
    """
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def tree_flatten_with_keys(self):
        keys = tuple(sorted(self))
        return tuple((jax.tree_util.DictKey(k), self[k]) for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


def random_layer_params(m, n, key, scale):
    """
    Initializes the parameters for a single layer of a neural network.
    Args:
        m: number of input features to the layer.
        n: number of output features from the layer.
        key: JAX random key for generating random numbers.
        scale: scaling factor for the weight initialization.
    """
    w_key, b_key = random.split(key, 2)
    return scale*random.normal(w_key, (m, n)), jnp.zeros(n)


def init_network_params(sizes, key):
    """
    Initializes the parameters for a multi-layer neural network.
    Args:
        sizes: a list of integers representing the number of neurons in each layer.
        key: JAX random key for generating random numbers.
    Returns:
        A list of tuples, where each tuple contains the weights and biases for a layer.
    """
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, 2.0/(jnp.sqrt(m+n))) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


@jax.jit
def init_train_state(config):
    """
    Initializes the training state for the neural network.
    Args:
        config: A configuration object containing the necessary parameters for initialization.
    Returns:
        A dot_dict containing the initialized training state.
    """
    train_state = dot_dict()
    train_state.params = init_param_state(config)
    train_state.opt_yield = init_adam_state(train_state.params.yield_func)
    train_state.opt_R_matrix = init_adam_state(train_state.params.R_matrix)
    train_state.lr_yield = config.learning_rate_initial_yield
    train_state.lr_R_matrix = config.learning_rate_initial
    train_state.norm = jnp.array(config.norm_state)
    return train_state


@jax.jit
def init_train_state_all(config):
    """
    Initializes the training state for the neural network without yield function optimization.
    Args:
        config: A configuration object containing the necessary parameters for initialization.
    Returns:
        A dot_dict containing the initialized training state.
    """
    train_state = dot_dict()
    train_state.params = init_param_state(config)
    train_state.opt = init_adam_state(train_state.params)
    train_state.lr = config.learning_rate_initial
    train_state.norm = jnp.array(config.norm_state)
    return train_state


@jax.jit
def init_train_state_pretraining(config):
    """
    Initializes the training state for the neural network with pretraining settings.
    Args:
        config: A configuration object containing the necessary parameters for initialization.
    Returns:
        A dot_dict containing the initialized training state.
    """

    train_state = dot_dict()
    train_state.params = init_param_state(config)
    train_state.opt_yield = init_adam_state(train_state.params.yield_func)
    train_state.opt = init_adam_state(train_state.params)
    train_state.lr_yield = config.learning_rate_initial_yield
    train_state.lr = config.learning_rate_initial
    train_state.norm = jnp.array(config.norm_state)
    return train_state


def init_param_state(config):
    """
    Initializes the parameter state for the neural network.
    Args:
        config: A configuration object containing the necessary parameters for initialization.
    Returns:
        A dot_dict containing the initialized parameter state.
    """
    root_key = jax.random.key(config.seed)
    key = map(partial(jax.random.fold_in, root_key), it.count())

    params = dot_dict()

    params.R_matrix = init_network_params(config.R_matrix_shape, next(key))
    if config.conservative_term:
        params.J_matrix = init_network_params(config.J_matrix_shape, next(key))
    
    if config.solver == "implicit_euler":
        if config.yield_func_type != "composed":
            params.yield_func = dot_dict(
                states = init_network_params(config.yield_func_states_shape, next(key)),
            )
            if config.yield_func_type == "asymp": 
                params.yield_func.bias = jnp.array(10.0)

        else:
            params.yield_func = dot_dict(
                smooth = init_network_params(config.yield_func_smooth_shape, next(key)),
                pos = init_network_params(config.yield_func_pos_shape, next(key)),
                strength = init_network_params(config.yield_func_strength_shape, next(key)),
            )

    return params


def init_adam_state(params):
    """
    Initializes the state for the Adam optimizer.
    Args:
        params: parameters of the neural network for which the Adam optimizer state is to be initialized.
    Returns:
        A dot_dict containing the initialized Adam optimizer state.
    """
    adam_state = dot_dict(mu = jax.tree.map(jnp.zeros_like, params), 
                          nu = jax.tree.map(jnp.zeros_like, params), 
                          count = jax.tree.map(lambda _: jnp.array(0), params))
    return adam_state


def adam_update(config, param, grad, adam_state, lr):
    """
    Computes the parameter update using the Adam optimization algorithm.
    This function updates the parameters of a neural network based on the gradients of the loss function with respect to the parameters, using the Adam optimization algorithm. It maintains moving averages of the gradients and their
    Args:
        config: A configuration object containing the necessary parameters for the Adam optimizer.
        param: The current parameters of the neural network.
        grad: The gradients of the loss with respect to the parameters.
        adam_state: The current state of the Adam optimizer.
        lr: The learning rate.
    Returns:
        The updated parameters and the updated Adam optimizer state.
    """

    new_mu = jax.tree.map(lambda x, y: config.beta_1 * x + (1 - config.beta_1) * y, adam_state.mu, grad)
    new_nu = jax.tree.map(lambda x, y: config.beta_2 * x + (1 - config.beta_2) * y**2, adam_state.nu, grad)
    new_count = jax.tree.map(lambda x: x + 1, adam_state.count)

    mu_hat = jax.tree.map(lambda m, c: m / (1 - config.beta_1 ** c), new_mu, new_count)
    nu_hat = jax.tree.map(lambda n, c: n / (1 - config.beta_2 ** c), new_nu, new_count)
    new_param = jax.tree.map(lambda p, m, n: p - lr * (m / (jnp.sqrt(n + config.eps_root) + config.eps)), param, mu_hat, nu_hat)
    
    new_adam_state = dot_dict(mu=new_mu, nu=new_nu, count=new_count)
    return new_param, new_adam_state


def prefetch_to_device(batches, size=2):
    """
    Prefetches batches of data to the device (e.g., GPU) for faster training.
    Args:
        batches: an iterable of batches of data.
        size: the number of batches to prefetch to the device.
    Yields:
        Batches of data prefetched to the device.
    """

    if size < 1:
        raise ValueError("Prefetch size must be at least one batch.")

    iterator = iter(batches)
    queue = deque()
    for _ in range(size):
        try:
            queue.append(jax.device_put(next(iterator)))
        except StopIteration:
            break

    while queue:
        batch = queue.popleft()
        try:
            queue.append(jax.device_put(next(iterator)))
        except StopIteration:
            pass
        yield batch


def save_checkpoint(path, **values):
    """
    Saves a checkpoint of the current state to a file.
    Args:
        path: The path to the checkpoint file.
        **values: The values to save in the checkpoint.
    """
    with path.open("wb") as checkpoint_file:
        pickle.dump(values, checkpoint_file)


def load_checkpoint(path):
    """
    Loads a checkpoint from a file.
    Args:
        path: The path to the checkpoint file.
    Returns:
        The loaded checkpoint.
    """
    with path.open("rb") as checkpoint_file:
        return pickle.load(checkpoint_file)


def load_dataset(config, split):
    """
    Loads a dataset from an HDF5 file and returns an iterator for the dataset.
    Args:
        config: configuration object containing dataset parameters.
        split: string indicating the dataset split (e.g., "train", "val", "test").
    Returns:
        An iterator for the dataset.
    """

    import h5py

    if split == "train" and (config.dataset == "long" or config.dataset == "mixed"):
        file = "train_l.h5"
    elif split == "train" and config.dataset == "all":
        file = "train_all.h5"
    else:
        file = f"{split}.h5"
    path = Path(config.data_folder) / file

    with h5py.File(path, "r") as data_file:
        input_chars = np.asarray(data_file["input_chars"]).T
        targets = np.asarray(data_file["targets"]).T
        states = np.asarray(data_file["states"]).T

    
    return DictIterator(
        config,
        input_chars,
        targets,
        states,
        split = split
    )


def loading_inputs(config, input_char, num_steps):
    """
    Construct the amplitude/angle loading ramp on the normalized interval [0, 1].
    Args:
        config: configuration object containing dataset parameters.
        input_char: a tuple containing the amplitude and angle of the loading ramp.
        num_steps: the number of steps in the loading ramp.

    Returns:
        A tensor containing the loading ramp.
    """
    t = jnp.linspace(0.0, num_steps * config.Dt, num_steps, dtype=config.dtype)
    angle = jnp.deg2rad(input_char[1])
    direction = jnp.array([jnp.cos(angle), jnp.sin(angle) / config.h])
    der_direction = jnp.array([-jnp.sin(angle), jnp.cos(angle)])

    inpt = input_char[0] * t[:, None] * direction
    der_inpt = input_char[0] * jnp.ones_like(t)[:, None] * der_direction
    return jnp.concatenate([inpt, der_inpt], axis=-1)

def load_yield_func_dataset(config, split):
    """
    Loads the yield function dataset from an HDF5 file and returns an iterator for the dataset.
    Args:
        config: configuration object containing dataset parameters.
        split: string indicating the dataset split (e.g., "train", "val", "test").
    Returns:
        An iterator for the yield function dataset.
    """

    import h5py

    if split == "train" and config.dataset == "long":
        file = "yield_surface_train_l.h5"
    elif split == "train" and (config.dataset == "all" or config.dataset == "mixed"):
        file = "yield_surface_train_all.h5"
    else:
        file = f"yield_surface_{split}.h5"
    path = Path(config.data_folder) / file

    with h5py.File(path, "r") as data_file:
        data = data_file["yield_surface"]
        states = np.concatenate([np.asarray(data["epsilon_p_bar"])[:, None], np.asarray(data["kappa_p_bar"])[:, None]], axis=-1)
        inputs = np.concatenate([np.asarray(data["epsilon_bar"])[:, None], np.asarray(data["kappa_bar"])[:, None]], axis=-1)
        targets_phi_z = np.asarray(data["phi_z"])

    return YieldDictIterator(
        config,
        states,
        inputs,
        targets_phi_z,
        split = split
    )