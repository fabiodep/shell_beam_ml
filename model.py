import jax
import jax.numpy as jnp

from jax import jit, vmap, custom_vjp, jacfwd, vjp
from jax.nn import leaky_relu, relu, softplus
from jax.lax import scan, cond
from functools import partial

from utils import newton_solve, trace, no_dev_dc_no_dev, no_dev_dc_dev, dev_dc_dev

def fnn(params, inpt, activation):
    """
    Feedforward neural network with arbitrary number of layers and activation function.
    Args:
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        inpt: input to the neural network.
        activation: activation function to be applied after each layer (except the last one).
    Returns:
        Output of the neural network after applying all layers and activation functions.
    """
    out = inpt
    for W, b in params[:-1]:
        out = activation(jnp.dot(out, W) + b)
    W, b = params[-1]
    return jnp.dot(out, W) + b


def R_matrix_block(config, params, inpt, activation):
    """
    Computes the R matrix block used for imposing the symmetric structure on the neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        inpt: input to the neural network.
        activation: activation function to be applied after each layer (except the last one).
    Returns:
        The computed R matrix block.
    """
    if config.diagonal_R:
        if config.system == "beam":
            R_matrix_ph = jnp.diag(fnn(params, inpt, activation).squeeze())
        elif config.system == "shell":
            R_matrix_ph = jnp.zeros((config.latent_dim, config.latent_dim))

            eps_bar = inpt[config.latent_dim:3*config.latent_dim//2]
            kappa_bar = inpt[3*config.latent_dim//2:]
            eps_p_bar = inpt[:config.latent_dim//2]
            kappa_p_bar = inpt[config.latent_dim//2:config.latent_dim]

            inpt_1 = trace(eps_bar)
            inpt_2 = trace(kappa_bar)
            inpt_3 = no_dev_dc_no_dev(eps_bar, eps_bar)
            inpt_4 = no_dev_dc_no_dev(kappa_bar, kappa_bar)
            inpt_5 = dev_dc_dev(eps_p_bar, eps_p_bar)
            inpt_6 = dev_dc_dev(kappa_p_bar, kappa_p_bar)
            inpt_7 = no_dev_dc_no_dev(eps_bar, kappa_bar)
            inpt_8 = no_dev_dc_dev(eps_bar, eps_p_bar)
            inpt_9 = no_dev_dc_dev(eps_bar, kappa_p_bar)
            inpt_10 = no_dev_dc_dev(kappa_bar, eps_p_bar)
            inpt_11 = no_dev_dc_dev(kappa_bar, kappa_p_bar)
            inpt_12 = dev_dc_dev(eps_p_bar, kappa_p_bar)

            inpt_nn = jnp.array([inpt_1, inpt_2, inpt_3, inpt_4, inpt_5, inpt_6, inpt_7, inpt_8, inpt_9, inpt_10, inpt_11, inpt_12])
            out = softplus(fnn(params, inpt_nn, activation).squeeze())

            dim = 4
            L_matrix_ph_eps = jnp.zeros((dim, dim))
            L_matrix_ph_kappa = jnp.zeros((dim, dim))

            pos = 0
            for i in range(dim):
                L_matrix_ph_eps = L_matrix_ph_eps.at[i, i:].set(out[pos:pos+(dim-i)])
                L_matrix_ph_kappa = L_matrix_ph_kappa.at[i, i:].set(out[pos+dim:pos+(dim-i)+dim])
                pos += dim - i

            eps_bar_dev = eps_bar - jnp.array([inpt_1 / 3, inpt_1 / 3, 0])
            kappa_bar_dev = kappa_bar - jnp.array([inpt_2 / 3, inpt_2 / 3, 0])
            basis = jnp.concatenate([eps_bar_dev[:, None], kappa_bar_dev[:, None], eps_p_bar[:, None], kappa_p_bar[:, None]], axis=1)

            R_matrix_ph = R_matrix_ph.at[:config.latent_dim//2, :config.latent_dim//2].set(basis @ L_matrix_ph_eps @ L_matrix_ph_eps.T @ basis.T)
            R_matrix_ph = R_matrix_ph.at[config.latent_dim//2:, config.latent_dim//2:].set(basis @ L_matrix_ph_kappa @ L_matrix_ph_kappa.T @ basis.T)
    else:
        R_matrix_ph = jnp.zeros((config.latent_dim, config.latent_dim))    
        R_matrix_entries = fnn(params, inpt, activation)
        pos = 0
        for i in range(config.latent_dim):
            R_matrix_ph = R_matrix_ph.at[i, i:].set(R_matrix_entries[pos:pos+(config.latent_dim-i)])
            pos += config.latent_dim - i
    return R_matrix_ph @ R_matrix_ph.T


def R_matrix(config, params, inpt):
    """ Computes the R matrix used for imposing the symmetric structure and the dissipative dynamics on the neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        inpt: input to the neural network.
    Returns:
        The computed R matrix.
    """
    if config.system == "beam" and config.symmetric:
        R_matrix_ph_pp = R_matrix_block(config, params, inpt, leaky_relu) 
        R_matrix_ph_mm = R_matrix_block(config, params, -1 * inpt, leaky_relu)
        R_matrix_ph_pm = R_matrix_block(config, params, jnp.concatenate([inpt[0][None], -1 * inpt[1][None], inpt[2][None], -1 * inpt[3][None]]), leaky_relu)
        R_matrix_ph_mp = R_matrix_block(config, params, jnp.concatenate([-1 * inpt[0][None], inpt[1][None], -1 * inpt[2][None], inpt[3][None]]), leaky_relu)
        R_matrix_ph = (R_matrix_ph_pp + R_matrix_ph_mm + R_matrix_ph_pm + R_matrix_ph_mp) / 4
        return R_matrix_ph
    elif config.system == "shell" and config.symmetric:
        eps_bar = inpt[config.latent_dim:3*config.latent_dim//2]
        kappa_bar = inpt[3*config.latent_dim//2:]
        eps_p_bar = inpt[:config.latent_dim//2]
        kappa_p_bar = inpt[config.latent_dim//2:config.latent_dim]

        R_matrix_pp = R_matrix_block(config, params, inpt, leaky_relu)
        R_matrix_pm = R_matrix_block(config, params, jnp.concatenate([eps_bar, -1 * kappa_bar, eps_p_bar, -1 * kappa_p_bar]), leaky_relu)
        R_matrix_mp = R_matrix_block(config, params, jnp.concatenate([-1 * eps_bar, kappa_bar, -1 * eps_p_bar, kappa_p_bar]), leaky_relu)
        R_matrix_mm = R_matrix_block(config, params, -1 * inpt, leaky_relu)

        return (R_matrix_pp + R_matrix_pm + R_matrix_mp + R_matrix_mm) / 4
    elif not config.symmetric:
        return R_matrix_block(config, params, inpt, leaky_relu)
    else:
        raise ValueError(f"Unknown configuration: system={config.system}, symmetric={config.symmetric}")


def J_matrix(config, params, inpt):
    """ 
    Computes the J matrix used for imposing the symmetric structure and the conservative dynamics on the neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        inpt: input to the neural network.
    Returns:
        The computed J matrix.
    """
    A_matrix_ph = jnp.zeros((config.latent_dim, config.latent_dim))
    A_matrix_entries = fnn(params, inpt, leaky_relu)
    pos = 0
    for i in range(config.latent_dim-1):
        A_matrix_ph = A_matrix_ph.at[(i+1):, i].set(A_matrix_entries[pos:pos+(config.latent_dim-1-i)])
        pos += config.latent_dim - 1 - i
    return A_matrix_ph - A_matrix_ph.T


def non_smooth_cond(x):
    """
    Returns a zero/one array indicating whether the input is close to zero.
    Args:
        x: input array.
    Returns:
        A zero/one array indicating whether the input is close to zero.
    """
    def true():
        return jnp.zeros_like(x[0])

    def false():
        return jnp.ones_like(x[0])

    return cond(jnp.isclose(jnp.linalg.norm(x), 0.0), true, false)


def phi_net_beam_complex(config, params, states, input, norm):
    """
    Computes the complex "hardening" function using a neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        states: current states of the system.
        input: input to the neural network.
        norm: normalization factor.
    Returns:
        The computed complex "hardening" function.  
    """
    if config.input_yield:
        normalized_inputs = input / norm
        normalized_states = states / norm
    else:
        normalized_inputs = input / norm
        normalized_states = states / norm
    if config.features:
        if config.features_type == "atan":
            features = jnp.atan(jnp.abs(normalized_states))
        elif config.features_type == "rs":
            features = jnp.sqrt(jnp.abs(normalized_states))
        elif config.features_type == "mixed":
            features = jnp.concatenate([jnp.sqrt(jnp.abs(normalized_states)), jnp.arctan(jnp.abs(normalized_states))])
        else:
            raise ValueError(f"Unknown features_type: {config.features_type}")

        net_pp = relu(non_smooth_cond(normalized_states) * fnn(params.states, jnp.concatenate([normalized_inputs, features]), leaky_relu).squeeze())
        net_mp = relu(non_smooth_cond(normalized_states) * fnn(params.states, jnp.concatenate([-1 * normalized_inputs[0][None], normalized_inputs[1][None], features]), leaky_relu).squeeze())
        net_mm = relu(non_smooth_cond(normalized_states) * fnn(params.states, jnp.concatenate([-1 * normalized_inputs, features]), leaky_relu).squeeze())
        net_pm = relu(non_smooth_cond(normalized_states) * fnn(params.states, jnp.concatenate([normalized_inputs[0][None], -1 * normalized_inputs[1][None], features]), leaky_relu).squeeze())
    else:
        net_pp = relu(non_smooth_cond(normalized_states) * fnn(params.states, normalized_inputs, leaky_relu).squeeze())
        net_mp = relu(non_smooth_cond(normalized_states) * fnn(params.states, jnp.concatenate([-1 * normalized_inputs[0][None], normalized_inputs[1][None]]), leaky_relu).squeeze())
        net_mm = relu(non_smooth_cond(normalized_states) * fnn(params.states, -1 * normalized_inputs, leaky_relu).squeeze())
        net_pm = relu(non_smooth_cond(normalized_states) * fnn(params.states, jnp.concatenate([normalized_inputs[0][None], -1 * normalized_inputs[1][None]]), leaky_relu).squeeze())
    return (net_pp + net_mp + net_mm + net_pm) / 4

def phi_net_beam(config, params, states, input, norm):
    """
    Computes the "hardening" function using a neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        states: current states of the system.
        input: input to the neural network.
        norm: normalization factor.
    Returns:
        The computed "hardening" function.
    """
    if config.input_yield:
        normalized_states = input / norm
    else:
        normalized_states = states / norm

    if config.symmetric:
        if config.symmetry_abs:
            normalized_states = jnp.abs(normalized_states)
            if config.root_square:
                sqrt_normalized_states = jnp.sqrt(normalized_states)
                net = relu(fnn(params.states, jnp.concatenate([normalized_states, sqrt_normalized_states]), leaky_relu).squeeze())
                return net
            else:
                net = relu(fnn(params.states, normalized_states, leaky_relu).squeeze())
                return net
        else:
            if config.features:
                if config.features_type == "atan":
                    features = jnp.atan(jnp.abs(normalized_states))
                elif config.features_type == "rs":
                    features = jnp.sqrt(jnp.abs(normalized_states))
                elif config.features_type == "mixed":
                    features = jnp.concatenate([jnp.sqrt(jnp.abs(normalized_states)), jnp.arctan(jnp.abs(normalized_states))])
                else:
                    raise ValueError(f"Unknown features_type: {config.features_type}")

                net_pp = relu(fnn(params.states, jnp.concatenate([normalized_states, features]), leaky_relu).squeeze())
                net_mp = relu(fnn(params.states, jnp.concatenate([-1 * normalized_states[0][None], normalized_states[1][None], features]), leaky_relu).squeeze())
                net_mm = relu(fnn(params.states, jnp.concatenate([-1 * normalized_states, features]), leaky_relu).squeeze())
                net_pm = relu(fnn(params.states, jnp.concatenate([normalized_states[0][None], -1 * normalized_states[1][None], features]), leaky_relu).squeeze())
            else:
                net_pp = relu(fnn(params.states, normalized_states, leaky_relu).squeeze())
                net_mp = relu(fnn(params.states, jnp.concatenate([-1 * normalized_states[0][None], normalized_states[1][None]]), leaky_relu).squeeze())
                net_mm = relu(fnn(params.states, -1 * normalized_states, leaky_relu).squeeze())
                net_pm = relu(fnn(params.states, jnp.concatenate([normalized_states[0][None], -1 * normalized_states[1][None]]), leaky_relu).squeeze())
            return (net_pp + net_mp + net_mm + net_pm) / 4
    else:
        return relu(fnn(params.states, normalized_states, leaky_relu).squeeze())

def asymptotic_value(input):
    """
    Computes the asymptotic value of the "hardening" function based on the input.
    Args:
        input: input to the neural network.
    Returns:
        The computed asymptotic value of the "hardening" function.
    """
    cot_theta = (input[0] / input[1])

    def true_N():
        return jnp.array(1.0)

    def false_N():
        return jnp.abs(2 * cot_theta)

    N_asym = cond(jnp.abs(cot_theta) >= 0.5, true_N, false_N)

    def true_M():
        return jnp.array(0.0)

    def false_M():
        return jnp.abs(6 * (0.25 - cot_theta**2))

    M_asym = cond(jnp.abs(cot_theta) >= 0.5, true_M, false_M)

    return N_asym + M_asym

def phi_asym_block(config, params, net_inputs, r, asymptotic_value):
    """ 
    Computes a block of the asymptotic "hardening" function using a neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        net_inputs: input to the neural network.
        r: norm of the normalized states.
        asymptotic_value: asymptotic value of the "hardening" function.
    Returns:
        The computed block of the asymptotic "hardening" function.
    """
    a = config.safety_eps + softplus(fnn(params.states, net_inputs, leaky_relu).squeeze())
    g = 1 - jnp.exp(-1 * r * a)
    return 1 - g * (asymptotic_value - 1)

def phi_net_beam_asym(config, params, states, input, norm):
    """ 
    Computes the asymptotic "hardening" function using a neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        states: current states of the system.
        input: input to the neural network.
        norm: normalization factor.
    Returns:
        The computed asymptotic "hardening" function.
    """
    normalized_states = states / norm
    r = jnp.linalg.norm(normalized_states)
    phi_z_asympt = asymptotic_value(input)
    phi_z_pp = phi_asym_block(config, params, normalized_states, r, phi_z_asympt)
    phi_z_mp = phi_asym_block(config, params, jnp.concatenate([-1 * normalized_states[0][None], normalized_states[1][None]]), r, phi_z_asympt)
    phi_z_mm = phi_asym_block(config, params, -1 * normalized_states, r, phi_z_asympt)
    phi_z_pm = phi_asym_block(config, params, jnp.concatenate([normalized_states[0][None], -1 * normalized_states[1][None]]), r, phi_z_asympt)

    #jax.debug.breakpoint(num_frames=1)
    return (phi_z_pp + phi_z_mp + phi_z_mm + phi_z_pm) / 4

def non_smooth_function(x):
    """ 
    Computes a non-smooth function used in the "hardening" function.
    Args:
        x: input to the function.
    Returns:
        The computed non-smooth function.
    """
    return 0.5 * (1 + jnp.sign(x) * (1 - jnp.exp(-1 * jnp.float_power(jnp.abs(x), 1.0 / 5))))


def phi_net_composed_block(config, params, normalized_states):
    """ 
    Computes a block of the composed "hardening" function using a neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        normalized_states: normalized states of the system.
    Returns:
        The computed block of the composed "hardening" function.
    """
    if config.features:
        if config.features_type == "atan":
            features = jnp.atan(jnp.abs(normalized_states))
        elif config.features_type == "rs":
            features = jnp.sqrt(jnp.abs(normalized_states))
        elif config.features_type == "mixed":
            features = jnp.concatenate([jnp.sqrt(jnp.abs(normalized_states)), jnp.arctan(jnp.abs(normalized_states))])
        else:
            raise ValueError(f"Unknown features_type: {config.features_type}")

        smooth = relu(fnn(params.smooth, jnp.concatenate([normalized_states, features]), leaky_relu).squeeze())
        strength = relu(fnn(params.strength, jnp.concatenate([normalized_states, features]), leaky_relu).squeeze())
        pos_effect = non_smooth_function(fnn(params.pos, jnp.concatenate([normalized_states, features]), leaky_relu).squeeze())
    else:
        smooth = relu(fnn(params.smooth, normalized_states, leaky_relu).squeeze())
        strength = relu(fnn(params.strength, normalized_states, leaky_relu).squeeze())
        pos_effect = non_smooth_function(fnn(params.pos, normalized_states, leaky_relu).squeeze())
    return smooth + strength * pos_effect

def phi_net_beam_composed(config, params, states, input, norm):
    """ 
    Computes the composed "hardening" function using a neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        states: current states of the system.
        input: input to the neural network.
        norm: normalization factor.
    Returns:
        The computed composed "hardening" function.
    """
    if config.input_yield:
        normalized_states = input / norm
    else:
        normalized_states = states / norm
    phi_z_pp = phi_net_composed_block(config, params, normalized_states)
    phi_z_mp = phi_net_composed_block(config, params, jnp.concatenate([-1 * normalized_states[0][None], normalized_states[1][None]]))
    phi_z_mm = phi_net_composed_block(config, params, -1 * normalized_states)
    phi_z_pm = phi_net_composed_block(config, params, jnp.concatenate([normalized_states[0][None], -1 * normalized_states[1][None]]))

    return (phi_z_pp + phi_z_mp + phi_z_mm + phi_z_pm) / 4

def start_yield_func_shell(config, N, M):
    """ 
    Computes the starting yield function for shell systems.
    Args:
        config: configuration object.
        N: axial force.
        M: bending moment.
    Returns:
        The computed starting yield function for shell systems.
    """
    f2 = (2 * (N[0]**2 + N[1]**2 - N[0] * N[1]) / 3 + N[2]**2) / config.h**2
    f2 += (36 * (M[0]**2 + M[1]**2 - M[0] * M[1]) + 54 * M[2]**2) / config.h**4
    f2 += 6 * jnp.abs(2 * N[0] * M[0] + 2 * N[1] * M[1] - N[0] * M[1] - N[1] * M[0] + 3 * N[2] * M[2]) / config.h**3
    return jnp.sqrt(f2) / config.sigma_y

def phi_net_shell(config, params, states, input, norm):
    """ 
    Computes the "hardening" function for shell systems using a neural network.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        states: current states of the system.
        input: input to the neural network.
        norm: normalization factor.
    Returns:
        The computed "hardening" function.
    """
    if config.input_yield:
        normalized_states = input / norm
    else:
        normalized_states = states / norm

    if config.symmetric and config.input_yield:
        eps_bar = input[:config.latent_dim//2]
        kappa_bar = input[config.latent_dim//2:config.latent_dim]
        inpt_1 = trace(eps_bar)
        inpt_2 = trace(kappa_bar)
        inpt_3 = no_dev_dc_no_dev(eps_bar, eps_bar)
        inpt_4 = no_dev_dc_no_dev(kappa_bar, kappa_bar)
        inpt_5 = no_dev_dc_no_dev(eps_bar, kappa_bar)

        phi_pp = relu(fnn(params.states, jnp.array([inpt_1, inpt_2, inpt_3, inpt_4, inpt_5]), leaky_relu).squeeze())
        phi_pm = relu(fnn(params.states, jnp.array([inpt_1, -1 * inpt_2, inpt_3, inpt_4, -1 * inpt_5]), leaky_relu).squeeze())
        phi_mp = relu(fnn(params.states, jnp.array([-1 * inpt_1, inpt_2, inpt_3, inpt_4, -1 * inpt_5]), leaky_relu).squeeze())
        phi_mm = relu(fnn(params.states, jnp.array([-1 * inpt_1, -1 * inpt_2, inpt_3, inpt_4, inpt_5]), leaky_relu).squeeze())

        return (phi_pp + phi_pm + phi_mp + phi_mm) / 4
    elif config.symmetric and not config.input_yield:
        eps_p_bar = states[:config.latent_dim//2]
        kappa_p_bar = states[config.latent_dim//2:config.latent_dim]
        inpt_1 = dev_dc_dev(eps_p_bar, eps_p_bar)
        inpt_2 = dev_dc_dev(kappa_p_bar, kappa_p_bar)
        inpt_3 = dev_dc_dev(eps_p_bar, kappa_p_bar)

        phi_p = relu(fnn(params.states, jnp.array([inpt_1, inpt_2, inpt_3]), leaky_relu).squeeze())
        phi_m = relu(fnn(params.states, jnp.array([inpt_1, inpt_2, -1 * inpt_3]), leaky_relu).squeeze())

        return (phi_p + phi_m) / 2
    else:
        return relu(fnn(params.states, normalized_states, leaky_relu).squeeze())


def phi_func(config, params, states, input, norm):
    """ 
    Computes the "hardening" function using a neural network correcting for the type of training selected in the configuration.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        states: current states of the system.
        input: input to the neural network.
        norm: normalization factor.
    Returns:
        The computed "hardening" function.
    """
    if config.system == "beam":
        if config.yield_func_type != "asymp":
            if config.yield_func_type == "single":
                phi_z_internal = phi_net_beam
            elif config.yield_func_type == "complex":
                phi_z_internal = phi_net_beam_complex
            elif config.yield_func_type == "composed":
                phi_z_internal = phi_net_beam_composed

            if config.yield_log_loss:
                return jnp.exp(phi_z_internal(config, params, states, input, norm))
            else:
                return 1 + phi_z_internal(config, params, states, input, norm)
        else:
            return phi_net_beam_asym(config, params, states, input, norm)
    elif config.system == "shell":
        return 1 + phi_net_shell(config, params, states, input, norm)
    else:
        raise ValueError(f"Unknown system: {config.system}")

phi_func_vmap = vmap(phi_func, in_axes=(None, None, 0, 0, None))

def yield_function_implicit(config, params, N, M, states, input, norm):
    """ 
    Computes the yield function used in the implicit solver.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        N: axial force.
        M: bending moment.
        states: current states of the system.
        input: input to the neural network.
        norm: normalization factor.
    Returns:
        The computed yield function.
    """
    if config.system == "beam":
        return jnp.abs(N) / config.N_y + jnp.abs(M) / config.M_y - phi_func(config, params, states, input[0:2], norm)
    elif config.system == "shell":
        return start_yield_func_shell(config, N, M) - phi_func(config, params, states, input[0:6], norm)

def state_dot_implicit(config, params, states, inpt, norm):
    """ Computes the state derivative used in the implicit solver.
    Args:
        config: configuration object.
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer.
        states: current states of the system.
        inpt: input to the neural network.
        norm: normalization factor.
    Returns:
        The computed state derivative.
    """
    normalized_states = states / norm
    if config.system == "beam":
        normalized_input = inpt[:2] / norm
        forces = jnp.diag(jnp.array([config.E * config.A, config.E * config.I])) @ (states - inpt[:2])
    elif config.system == "shell":
        normalized_input = inpt[:6] / norm
        forces_N = jnp.array(config.E_mandel) @ (states[:3] - inpt[:3]) * config.h
        forces_M = jnp.array(config.E_mandel) @ (states[3:6] - inpt[3:6]) * config.h**3 / 12
        forces = jnp.concatenate([forces_N, forces_M]) 
    else:
        raise ValueError(f"Unknown system: {config.system}")
    inpt_net = jnp.concatenate([normalized_states, normalized_input])
    R_matrix_ph = R_matrix(config, params, inpt_net)
    if config.conservative_term:
        J_matrix_ph = J_matrix(config, params, inpt_net)
        state_dot_ph = - (J_matrix_ph + R_matrix_ph) @ forces / jnp.linalg.norm(forces)
    else:
        state_dot_ph = - R_matrix_ph @ forces / jnp.linalg.norm(forces)
    return state_dot_ph

state_dot_implicit_vmap = vmap(state_dot_implicit, in_axes=(None, None, 0, 0, None))

def residual_implicit(z, config, params_R_matrix, params_yield_func, states, inpt, norm):
    """ 
    Computes the residual used in the Newton method for the implicit solver.
    Args:
        z: the state vector.
        config: configuration object.
        params_R_matrix: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the R matrix.
        params_yield_func: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the yield function.
        states: current states of the system.
        inpt: input to the neural network.
        norm: normalization factor.
    Returns:
        The computed residual.
    """
    residual_ph = jnp.zeros((states.shape[0] + 1, ))
    residual_ph = residual_ph.at[:-1].set(z[:-1] - states - z[-1] * state_dot_implicit(config, params_R_matrix, z[:-1], inpt, norm))
    if config.system == "beam":
        N = config.E * config.A * (inpt[0] - z[0])
        M = -1 * config.E * config.I * (inpt[1] - z[1])
    elif config.system == "shell":
        N = jnp.array(config.E_mandel) @ (inpt[:3] - z[:3]) * config.h
        M = -1 * jnp.array(config.E_mandel) @ (inpt[3:6] - z[3:6]) * config.h**3 / 12
    else:
        raise ValueError(f"Unknown system: {config.system}")
    residual_ph = residual_ph.at[-1].set(yield_function_implicit(config, params_yield_func, N, M, z[:-1], inpt, norm))
    return residual_ph

@custom_vjp
def state_step_implicit(config, params_R_matrix, params_yield_func, states, inpt, norm):
    """ 
    Computes the next state using an implicit solver.
    Args:
        config: configuration object.
        params_R_matrix: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the R matrix.
        params_yield_func: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the yield function.
        states: current states of the system.
        inpt: input to the neural network.
        norm: normalization factor.
    Returns:
        The next state computed using the implicit solver.
    """
    residual_fn = residual_implicit
    initial_condition = jnp.concatenate([states, config.Dt * jnp.ones(1)])
    z = newton_solve(config, residual_fn, initial_condition, params_R_matrix, params_yield_func, states, inpt, norm)
    return z

def state_step_implicit_fwd(config, params_R_matrix, params_yield_func, states, inpt, norm):
    """ 
    Forward pass for the custom VJP of the implicit state step.
    Args:
        config: configuration object.
        params_R_matrix: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the R matrix.
        params_yield_func: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the yield function.
        states: current states of the system.
        inpt: input to the neural network.
        norm: normalization factor.
    Returns:
        The next state computed using the implicit solver and the residual data for the backward pass.
    """
    residual_fn = residual_implicit
    initial_condition = jnp.concatenate([states, config.Dt * jnp.ones(1)])
    z = newton_solve(config, residual_fn, initial_condition, params_R_matrix, params_yield_func, states, inpt, norm)
    return z, (z, config, params_R_matrix, params_yield_func, states, inpt, norm)

def state_step_implicit_bwd(residual_data, g):
    """ 
    Backward pass for the custom VJP of the implicit state step.
    Args:
        residual_data: data from the forward pass needed for the backward pass.
        g: gradient of the output with respect to the next state.
    Returns:
        Gradients with respect to the inputs of the implicit state step.
    """
    states_new, config, params_R_matrix, params_yield_func, states, inpt, norm = residual_data

    def residual(z):
        return residual_implicit(z, config, params_R_matrix, params_yield_func, states, inpt, norm)

    J_R = jacfwd(residual)(states_new)
    lmbd = jnp.linalg.solve(J_R.T, g)

    grad_states = lmbd

    def residual_params_state(p):
        return residual_implicit(states_new, config, p, params_yield_func, states, inpt, norm)

    def residual_params_yield(p):
            return residual_implicit(states_new, config, params_R_matrix, p, states, inpt, norm)

    def residual_inpt(i):
        return residual_implicit(states_new, config, params_R_matrix, params_yield_func, states, i, norm)

    _, vjp_params_state = vjp(residual_params_state, params_R_matrix)
    _, vjp_params_yield = vjp(residual_params_yield, params_yield_func)
    _, vjp_inpt = vjp(residual_inpt, inpt)

    grad_params_R_matrix = vjp_params_state(-lmbd)[0]
    grad_params_yield = vjp_params_yield(-lmbd)[0]
    grad_inpt = vjp_inpt(-lmbd)[0]

    return None, grad_params_R_matrix, grad_params_yield, grad_states[:-1], grad_inpt, None

state_step_implicit.defvjp(state_step_implicit_fwd, state_step_implicit_bwd)

def state_step(config, params_R_matrix, params_yield_func, states, yield_val, inpt, norm):
    """ 
    Computes the next state based on the yield condition.
    Args:
        config: configuration object.
        params_R_matrix: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the R matrix.
        params_yield_func: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the yield function.
        states: current states of the system.
        yield_val: current yield value.
        inpt: input to the neural network.
        norm: normalization factor.
    Returns:
        The next state, forces, state derivative, Lagrange multiplier, and yield value.
    """
    if config.system == "beam":
        N_tr = config.E * config.A * (inpt[0] - states[0])
        M_tr = -1 * config.E * config.I * (inpt[1] - states[1])
        Phi_tr = jnp.abs(N_tr) / config.N_y + jnp.abs(M_tr) / config.M_y - yield_val
        #jax.debug.breakpoint(num_frames=1)
    elif config.system == "shell":
        N_tr = jnp.array(config.E_mandel) @ (inpt[:3] - states[:3]) * config.h
        M_tr = -1 * jnp.array(config.E_mandel) @ (inpt[3:6] - states[3:6]) * config.h**3 / 12
        Phi_tr = start_yield_func_shell(config, N_tr, M_tr) - yield_val
    else:
        raise ValueError(f"Unknown system: {config.system}")

    def true():
        if config.system == "beam":
            return states, jnp.array([N_tr, M_tr]), jnp.zeros_like(states), jnp.zeros((1,)), yield_val
        elif config.system == "shell":
            return states, jnp.concatenate([N_tr, M_tr]), jnp.zeros_like(states), jnp.zeros((1,)), yield_val
        else:
            raise ValueError(f"Unknown system: {config.system}")

    def false():
        if config.solver == "implicit_euler":
            if config.system == "beam":
                yield_val = phi_func(config, params_yield_func, states, inpt[:2], norm)
            elif config.system == "shell":
                yield_val = phi_func(config, params_yield_func, states, inpt[:6], norm)
            else:
                raise ValueError(f"Unknown system: {config.system}")
            states_new = state_step_implicit(config, params_R_matrix, params_yield_func, states, inpt, norm)
            states_dot = state_dot_implicit(config, params_R_matrix, states, inpt, norm)
            DLmb = states_new[-1]

        if config.system == "beam":
            forces_new = jnp.zeros(2)
            forces_new = forces_new.at[0].set(config.E * config.A * (inpt[0] - states_new[0]))
            forces_new = forces_new.at[1].set(-1 * config.E * config.I * (inpt[1] - states_new[1]))
        elif config.system == "shell":
            forces_new = jnp.zeros(6)
            forces_new = forces_new.at[:3].set(jnp.array(config.E_mandel) @ (inpt[:3] - states[:3]) * config.h)
            forces_new = forces_new.at[3:6].set(-1 * jnp.array(config.E_mandel) @ (inpt[3:6] - states_new[3:6]) * config.h**3 / 12)
        else:
            raise ValueError(f"Unknown system: {config.system}")

        return states_new[:-1], forces_new, states_dot, DLmb[None], yield_val

    return cond(Phi_tr < 0, true, false)

@partial(jit, static_argnames=["config"])
def states_trajectory(config, params_R_matrix, params_yield_func, inputs, norm):
    """ 
    Computes the trajectory of states over time based on the inputs.
    Args:
        config: configuration object.
        params_R_matrix: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the R matrix.
        params_yield_func: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer of the yield function.
        inputs: sequence of inputs over time.
        norm: normalization factor.
    Returns:
        The trajectory of states over time.
    """
    
    def scan_fn(states, inpt):
        states_new, forces_new, states_dot, DLmb, yield_val = state_step(config, params_R_matrix, params_yield_func, states[:-1], states[-1], inpt, norm)
        return jnp.concatenate([states_new, yield_val[None]]), jnp.concatenate([states_new, forces_new, states_dot, DLmb, yield_val[None]])
     
    _, trajectory = scan(scan_fn, jnp.concatenate([jnp.zeros((config.latent_dim,)), jnp.ones((1,))]), xs=inputs)
    return trajectory[:,:-1]

state_trajectory_vmap = vmap(states_trajectory, in_axes=(None, None, 0, None))