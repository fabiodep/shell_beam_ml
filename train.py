import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt

from dataclasses import replace
from functools import partial
from pathlib import Path
from tqdm import trange
from typing import Sequence, Mapping

from config import Config, load_config
from eval import eval_yield_func_fn, eval_state_func_fn
from loss import loss_direction_state_fn, loss_yield_func_fn, loss_state_func_fn, loss_direction_state_fn
from utils import *

@partial(jax.jit, static_argnames=["config"])
def train_step_yield_func(config, train_state, batch, step):
    """ 
    Performs a single training step for the yield function.
    Args:
        config: configuration object.
        train_state: current state of the training, including parameters and optimizers.
        batch: batch of data for training, containing states, inputs, and target yield values.
        step: current training step.
    Returns:
        new_train_state: updated training state after the step.
        train_loss: computed loss for the current batch.
    """

    def objective(params):
        return loss_yield_func_fn(config, params, batch["states"], batch["inputs"], batch["targets_phi_z"], train_state.norm)

    train_loss, grads = jax.value_and_grad(objective)(train_state.params.yield_func)
    new_params_yield_func, new_opt = adam_update(config, train_state.params.yield_func, grads, train_state.opt_yield, train_state.lr_yield)

    if config.high_and_low_lr:
        new_lr = jax.lax.select(
            step < config.high_steps,
            config.learning_rate_initial,
            config.learning_rate_low
        )
    else:
        decay = (step + 1) % config.lr_decay_steps == 0
        new_lr = jax.lax.select(
            decay,
            train_state.lr_yield * config.lr_decay_rate,
            train_state.lr_yield
        )

    new_params = train_state.params
    new_params.yield_func = new_params_yield_func
    if config.pretraining:
        new_train_state = dot_dict(
            params = new_params,
            opt_yield = new_opt,
            opt = train_state.opt,
            lr_yield = new_lr,
            lr = train_state.lr,
            norm = train_state.norm
        )
    else:
        new_train_state = dot_dict(
            params = new_params,
            opt_yield = new_opt,
            opt_R_matrix = train_state.opt_R_matrix,
            lr_yield = new_lr,
            lr_R_matrix = train_state.lr_R_matrix,
            norm = train_state.norm
        )

    #jax.debug.breakpoint()
    return new_train_state, train_loss

@partial(jax.jit, static_argnames=["config"])
def train_step_state_func(config, train_state, batch, step):
    """ 
    Performs a single training step for the state function.
    Args:
        config: configuration object.
        train_state: current state of the training, including parameters and optimizers.
        batch: batch of data for training, containing input characteristics, states, and target forces.
        step: current training step.
    Returns:
        new_train_state: updated training state after the step.
        train_loss: computed loss for the current batch.
    """

    def objective(params):
        if config.direction_loss:
            return loss_direction_state_fn(config, params, batch["input_chars"], batch["states"], train_state.norm)
        else:
            return loss_state_func_fn(config, params, train_state.params.yield_func, batch["input_chars"], batch["targets"], batch["states"], train_state.norm)

    train_loss, grads = jax.value_and_grad(objective)(train_state.params.R_matrix)
    new_params_R_matrix, new_opt_R_matrix = adam_update(config, train_state.params.R_matrix, grads, train_state.opt_R_matrix, train_state.lr_R_matrix)

    decay = (step + 1) % config.lr_decay_steps == 0
    new_lr = jax.lax.select(
        decay,
        train_state.lr_R_matrix * config.lr_decay_rate,
        train_state.lr_R_matrix
    )

    new_params = train_state.params
    new_params.R_matrix = new_params_R_matrix
    new_train_state = dot_dict(
        params = new_params,
        opt_yield = train_state.opt_yield,
        opt_R_matrix = new_opt_R_matrix,
        lr_yield = train_state.lr_yield,
        lr_R_matrix = new_lr,
        norm = train_state.norm
    )

    #jax.debug.breakpoint()
    return new_train_state, train_loss


@partial(jax.jit, static_argnames=["config"])
def train_step(config, train_state, batch, step):
    """ 
    Performs a single training step for both the yield function and the state function.
    Args:
        config: configuration object.
        train_state: current state of the training, including parameters and optimizers.
        batch: batch of data for training, containing input characteristics, states, and target forces.
        step: current training step.
    Returns:
        new_train_state: updated training state after the step.
        train_loss: computed loss for the current batch.
    """
    def objective(params):
        return loss_state_func_fn(config, params.R_matrix, params.yield_func, batch["input_chars"], batch["targets"], batch["states"], train_state.norm)

    train_loss, grads = jax.value_and_grad(objective)(train_state.params)
    new_params, new_opt = adam_update(config, train_state.params, grads, train_state.opt, train_state.lr)

    decay = (step + 1) % config.lr_decay_steps == 0
    new_lr = jax.lax.select(
        decay,
        train_state.lr * config.lr_decay_rate,
        train_state.lr
    )

    if config.yield_params_dumping:
        yield_params_old = train_state.params.yield_func
        yield_params_new = new_params.yield_func
        yield_params_new = jax.tree_util.tree_map(
            lambda old, new: old * (1 - config.yield_params_dumping_factor) + new * config.yield_params_dumping_factor,
            yield_params_old,
            yield_params_new
        )
        new_params.yield_func = yield_params_new

    if config.pretraining:
        new_train_state = dot_dict(
            params = new_params,
            opt_yield = train_state.opt_yield,
            opt = new_opt,
            lr_yield = train_state.lr_yield,
            lr = new_lr,
            norm = train_state.norm
        )
    else:
        new_train_state = dot_dict(
            params = new_params,
            opt = new_opt,
            lr = new_lr,
            norm = train_state.norm
        )

    #jax.debug.breakpoint()
    return new_train_state, train_loss


def train_loop_yield_func(
    config: Config | None = None,
    resume_from: Path | None = None,
) -> None:
    """ 
    Runs the training loop for the yield function.
    Args:
        config: configuration object.
        resume_from: path to a checkpoint to resume training from.
        config_overrides: dictionary of configuration parameters to override.
    """
    
    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        checkpoint = load_checkpoint(resume_from)
        config = load_config(resume_from.parent)
        train_state = jax.device_put(checkpoint["train_state"])
        step = jax.device_put(checkpoint["step"])
        start_epoch = int(checkpoint["epoch_yield"]) + 1
        yield_func_losses = list(checkpoint.get("yield_func_losses", ()))
        state_func_losses = list(checkpoint.get("state_func_losses", ()))
        mean_train_L2_errors = list(checkpoint.get("mean_train_L2_errors", ()))
        mean_val_L2_errors = list(checkpoint.get("mean_val_L2_errors", ()))
        train_yield_func_MSE = list(checkpoint.get("train_yield_func_MSE", ()))
        val_yield_func_MSE = list(checkpoint.get("val_yield_func_MSE", ()))
        evaluation_epochs = list(checkpoint.get("evaluation_epochs_yield", ()))
        print(f"Resuming from {resume_from} after epoch {start_epoch-1} ...")
    else:
        start_epoch = 0
        if config.pretraining:
            train_state = jax.device_put(init_train_state_pretraining(config))
        else:
            train_state = jax.device_put(init_train_state(config))
        step = jax.device_put(jnp.array(0))
        yield_func_losses = []
        state_func_losses = []
        mean_train_L2_errors = []
        mean_val_L2_errors = []
        train_yield_func_MSE = []
        val_yield_func_MSE = []
        evaluation_epochs = []

    train_dataset = load_yield_func_dataset(config, "train")
    val_dataset = load_yield_func_dataset(config, "val")
    
    save_path = Path(config.save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    print("Training yield_surface ...")

    for epoch in trange(start_epoch, config.num_epochs_yield):
        train_dataset.reset()
        epoch_loss = jnp.array(0.0, dtype=config.dtype)
        sample_count = 0
        previous_params = train_state.params

        for batch in prefetch_to_device(train_dataset, config.prefetch_batches):
            train_state, batch_loss = train_step_yield_func(config, train_state, batch, step)
            batch_size = batch["states"].shape[0]
            epoch_loss += batch_loss * batch_size
            sample_count += batch_size
            step += 1
        yield_func_losses.append(float(epoch_loss / sample_count))

        evaluate_epoch = epoch == 0 or epoch == config.num_epochs_yield - 1
        evaluate_epoch |= epoch % config.eval_interval_yield == 0
        if evaluate_epoch:
            train_MSE = eval_yield_func_fn(config, train_state.params, train_dataset, train_state.norm)
            val_MSE = eval_yield_func_fn(config, train_state.params, val_dataset, train_state.norm)

            evaluation_epochs.append(epoch)
            train_yield_func_MSE.append(float(train_MSE))
            val_yield_func_MSE.append(float(val_MSE))

            save_yield_func_history_plots(config, 
                                          yield_func_losses, 
                                          evaluation_epochs, 
                                          train_yield_func_MSE,
                                          val_yield_func_MSE)



        save_epoch = epoch == 0 or epoch % config.save_interval == 0
        if save_epoch:
            save_checkpoint(
                save_path / "model_checkpoint.pkl",
                step = step,
                epoch_yield = epoch,
                epoch = 0,
                train_state = train_state,
                previous_params = previous_params,
                yield_func_losses = yield_func_losses,
                state_func_losses = state_func_losses,
                mean_train_L2_errors = mean_train_L2_errors,
                mean_val_L2_errors = mean_val_L2_errors,
                train_yield_func_MSE = train_yield_func_MSE,
                val_yield_func_MSE = val_yield_func_MSE,
                evaluation_epochs = evaluation_epochs
            )
        if epoch == config.num_epochs_yield - 1:
            save_checkpoint(
                save_path / "model.pkl",
                step = step,
                epoch_yield = epoch,
                epoch = 0,
                train_state = train_state,
                previous_params = previous_params,
                yield_func_losses = yield_func_losses,
                state_func_losses = state_func_losses,
                mean_train_L2_errors = mean_train_L2_errors,
                mean_val_L2_errors = mean_val_L2_errors,
                train_yield_func_MSE = train_yield_func_MSE,
                val_yield_func_MSE = val_yield_func_MSE,
                evaluation_epochs = []
            )


def train_loop_state_func(
    config: Config | None = None,
    resume_from: Path | None = None,
) -> None:
    """ 
    Runs the training loop for the state function.
    Args:
        config: configuration object.
        resume_from: path to a checkpoint to resume training from.
        config_overrides: dictionary of configuration parameters to override.
    """

    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        checkpoint = load_checkpoint(resume_from)
        config = load_config(resume_from.parent)
        train_state = jax.device_put(checkpoint["train_state"])
        step = jax.device_put(checkpoint["step"])
        previous_params = jax.device_put(checkpoint["previous_params"])
        start_epoch = int(checkpoint["epoch"]) + 1
        yield_func_losses = list(checkpoint.get("yield_func_losses", ()))
        state_func_losses = list(checkpoint.get("state_func_losses", ()))
        train_yield_func_MSE = list(checkpoint.get("train_yield_func_MSE", ()))
        val_yield_func_MSE = list(checkpoint.get("val_yield_func_MSE", ()))
        mean_train_L2_errors = list(checkpoint.get("mean_train_L2_errors", ()))
        mean_val_L2_errors = list(checkpoint.get("mean_val_L2_errors", ()))
        evaluation_epochs = list(checkpoint.get("evaluation_epochs", ()))
        print(f"Resuming from {resume_from} after epoch {start_epoch-1} ...")
    else:
        resume_from = Path(config.save_folder) / "model.pkl"
        checkpoint = load_checkpoint(resume_from)
        start_epoch = 0
        train_state = jax.device_put(checkpoint["train_state"])
        step = jax.device_put(jnp.array(0))
        yield_func_losses = list(checkpoint.get("yield_func_losses", ()))
        state_func_losses = []
        train_yield_func_MSE = list(checkpoint.get("train_yield_func_MSE", ()))
        val_yield_func_MSE = list(checkpoint.get("val_yield_func_MSE", ()))
        mean_train_L2_errors = []
        mean_val_L2_errors = []
        evaluation_epochs = []

    train_dataset = load_dataset(config, "train")
    val_dataset = load_dataset(config, "val")

    
    save_path = Path(config.save_folder)

    print("Training state function ...")

    for epoch in trange(start_epoch, config.num_epochs):
        train_dataset.reset()
        epoch_loss = jnp.array(0.0, dtype=config.dtype)
        sample_count = 0
        previous_params = train_state.params

        for batch in prefetch_to_device(train_dataset, config.prefetch_batches):
            train_state, batch_loss = train_step_state_func(config, train_state, batch, step)
            batch_size = batch["targets"].shape[0]
            epoch_loss += batch_loss * batch_size
            sample_count += batch_size
            step += 1
        state_func_losses.append(float(epoch_loss / sample_count))

        evaluate_epoch = epoch == 0 or epoch == config.num_epochs - 1
        evaluate_epoch |= epoch % config.eval_interval_state == 0
        if evaluate_epoch:
            train_L2_errors, train_forces, train_targets, train_angles = eval_state_func_fn(config,
                                                                                            train_state.params,
                                                                                            train_dataset,
                                                                                            train_state.norm)
            val_L2_errors, val_forces, val_targets, val_angles = eval_state_func_fn(config,
                                                                                    train_state.params,
                                                                                    val_dataset,
                                                                                    train_state.norm)

            evaluation_epochs.append(epoch)
            mean_train_L2_errors.append(float(jnp.mean(train_L2_errors)))
            mean_val_L2_errors.append(float(jnp.mean(val_L2_errors)))

            save_history_plots(config, 
                               state_func_losses, 
                               evaluation_epochs, 
                               mean_train_L2_errors,
                               mean_val_L2_errors)

            save_loads_plots(config, epoch, "train", train_L2_errors, train_forces, train_targets, train_angles)
            save_loads_plots(config, epoch, "val", val_L2_errors, val_forces, val_targets, val_angles)

        save_epoch = epoch == 0 or epoch % config.save_interval == 0
        if save_epoch:
            save_checkpoint(
                save_path / "model_checkpoint.pkl",
                step = step,
                epoch_yield = checkpoint["epoch_yield"],
                epoch = epoch,
                train_state = train_state,
                previous_params = previous_params,
                yield_func_losses = yield_func_losses,
                state_func_losses = state_func_losses,
                mean_train_L2_errors = mean_train_L2_errors,
                mean_val_L2_errors = mean_val_L2_errors,
                train_yield_func_MSE = train_yield_func_MSE,
                val_yield_func_MSE = val_yield_func_MSE,
                evaluation_epochs = evaluation_epochs
            )
        if epoch == config.num_epochs - 1:
            save_checkpoint(
                save_path / "model.pkl",
                step = step,
                epoch_yield = checkpoint["epoch_yield"],
                epoch = epoch,
                train_state = train_state,
                previous_params = previous_params,
                yield_func_losses = yield_func_losses,
                state_func_losses = state_func_losses,
                mean_train_L2_errors = mean_train_L2_errors,
                mean_val_L2_errors = mean_val_L2_errors,
                train_yield_func_MSE = train_yield_func_MSE,
                val_yield_func_MSE = val_yield_func_MSE,
                evaluation_epochs = evaluation_epochs
            )


def train_loop(
    config: Config | None = None,
    resume_from: Path | None = None,
) -> None:
    """ Runs the training loop for both the yield function and the state function.
    Args:
        config: configuration object.
        resume_from: path to a checkpoint to resume training from.
        config_overrides: dictionary of configuration parameters to override.
    """

    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        checkpoint = load_checkpoint(resume_from)
        config = load_config(resume_from.parent)
        train_state = jax.device_put(checkpoint["train_state"])
        step = jax.device_put(checkpoint["step"])
        previous_params = jax.device_put(checkpoint["previous_params"])
        start_epoch = int(checkpoint["epoch"]) + 1
        state_func_losses = list(checkpoint.get("state_func_losses", ()))
        mean_train_L2_errors = list(checkpoint.get("mean_train_L2_errors", ()))
        mean_val_L2_errors = list(checkpoint.get("mean_val_L2_errors", ()))
        evaluation_epochs = list(checkpoint.get("evaluation_epochs", ()))
        if config.pretraining:
            yield_func_losses = list(checkpoint.get("yield_func_losses", ()))
            train_yield_func_MSE = list(checkpoint.get("train_yield_func_MSE", ()))
            val_yield_func_MSE = list(checkpoint.get("val_yield_func_MSE", ()))
        print(f"Resuming from {resume_from} after epoch {start_epoch-1} ...")
    else:
        if config.pretraining:
            resume_from = Path(config.save_folder) / "model.pkl"
            checkpoint = load_checkpoint(resume_from)
            start_epoch = 0
            train_state = jax.device_put(checkpoint["train_state"])
            step = jax.device_put(jnp.array(0))
            yield_func_losses = list(checkpoint.get("yield_func_losses", ()))
            state_func_losses = []
            train_yield_func_MSE = list(checkpoint.get("train_yield_func_MSE", ()))
            val_yield_func_MSE = list(checkpoint.get("val_yield_func_MSE", ()))
            mean_train_L2_errors = []
            mean_val_L2_errors = []
            evaluation_epochs = []
        else:
            start_epoch = 0
            train_state = jax.device_put(init_train_state_all(config))
            step = jax.device_put(jnp.array(0))
            state_func_losses = []
            mean_train_L2_errors = []
            mean_val_L2_errors = []
            evaluation_epochs = []

    train_dataset = load_dataset(config, "train")
    val_dataset = load_dataset(config, "val")
    
    save_path = Path(config.save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    print("Training ...")

    for epoch in trange(start_epoch, config.num_epochs):
        train_dataset.reset()
        epoch_loss = jnp.array(0.0, dtype=config.dtype)
        sample_count = 0
        previous_params = train_state.params

        for batch in prefetch_to_device(train_dataset, config.prefetch_batches):
            train_state, batch_loss = train_step(config, train_state, batch, step)
            batch_size = batch["targets"].shape[0]
            epoch_loss += batch_loss * batch_size
            sample_count += batch_size
            step += 1
        state_func_losses.append(float(epoch_loss / sample_count))

        evaluate_epoch = epoch == 0 or epoch == config.num_epochs - 1
        evaluate_epoch |= epoch % config.eval_interval_state == 0
        if evaluate_epoch:
            train_L2_errors, train_forces, train_targets, train_angles = eval_state_func_fn(config,
                                                                                            train_state.params,
                                                                                            train_dataset,
                                                                                            train_state.norm)
            val_L2_errors, val_forces, val_targets, val_angles = eval_state_func_fn(config,
                                                                                    train_state.params,
                                                                                    val_dataset,
                                                                                    train_state.norm)

            evaluation_epochs.append(epoch)
            mean_train_L2_errors.append(float(jnp.mean(train_L2_errors)))
            mean_val_L2_errors.append(float(jnp.mean(val_L2_errors)))

            save_history_plots(config, 
                               state_func_losses, 
                               evaluation_epochs, 
                               mean_train_L2_errors,
                               mean_val_L2_errors,)

            save_loads_plots(config, epoch, "train", train_L2_errors, train_forces, train_targets, train_angles)
            save_loads_plots(config, epoch, "val", val_L2_errors, val_forces, val_targets, val_angles)

        save_epoch = epoch == 0 or epoch % config.save_interval == 0
        if save_epoch:
            if config.pretraining:
                save_checkpoint(
                    save_path / "model_checkpoint.pkl",
                    step = step,
                    epoch = epoch,
                    train_state = train_state,
                    previous_params = previous_params,
                    yield_func_losses = yield_func_losses,
                    state_func_losses = state_func_losses,
                    mean_train_L2_errors = mean_train_L2_errors,
                    mean_val_L2_errors = mean_val_L2_errors,
                    train_yield_func_MSE = train_yield_func_MSE,
                    val_yield_func_MSE = val_yield_func_MSE,
                    evaluation_epochs = evaluation_epochs
                )
            else:
                save_checkpoint(
                    save_path / "model_checkpoint.pkl",
                    step = step,
                    epoch = epoch,
                    train_state = train_state,
                    previous_params = previous_params,
                    state_func_losses = state_func_losses,
                    mean_train_L2_errors = mean_train_L2_errors,
                    mean_val_L2_errors = mean_val_L2_errors,
                    evaluation_epochs = evaluation_epochs
                )
        if epoch == config.num_epochs - 1:
            if config.pretraining:
                save_checkpoint(
                    save_path / "model.pkl",
                    step = step,
                    epoch = epoch,
                    train_state = train_state,
                    previous_params = previous_params,
                    yield_func_losses = yield_func_losses,
                    state_func_losses = state_func_losses,
                    mean_train_L2_errors = mean_train_L2_errors,
                    mean_val_L2_errors = mean_val_L2_errors,
                    train_yield_func_MSE = train_yield_func_MSE,
                    val_yield_func_MSE = val_yield_func_MSE,
                    evaluation_epochs = evaluation_epochs
                )
            else:
                save_checkpoint(
                    save_path / "model.pkl",
                    step = step,
                    epoch = epoch,
                    train_state = train_state,
                    previous_params = previous_params,
                    yield_func_losses = yield_func_losses,
                    state_func_losses = state_func_losses,
                    mean_train_L2_errors = mean_train_L2_errors,
                    mean_val_L2_errors = mean_val_L2_errors,
                    evaluation_epochs = evaluation_epochs
                )


def save_yield_func_history_plots(config, 
                                  losses, 
                                  evaluation_epochs, 
                                  train_MSE,
                                  val_MSE):
    """ 
    Saves the training history plots for the yield function.
    Args:
        config: configuration object.
        losses: list of training losses over epochs.
        evaluation_epochs: list of epochs at which evaluations were performed.
        train_MSE: list of training mean squared errors over evaluation epochs.
        val_MSE: list of validation mean squared errors over evaluation epochs.
    """

    save_path = Path(config.save_folder)

    plt.figure(figsize=(8, 6))
    plt.semilogy(losses)
    plt.xlabel("Training epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.tight_layout()
    plt.title(f"Loss (final = {losses[-1]:.4g})")
    plt.savefig(save_path / "loss_yield_func.pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.semilogy(evaluation_epochs, train_MSE, label="Train MSE")
    plt.semilogy(evaluation_epochs, val_MSE, label="Validation MSE")
    plt.xlabel("Training epochs")
    plt.ylabel("MSE")
    plt.title(f"MSE (train = {train_MSE[-1]:.4g}, val = {val_MSE[-1]:.4g})")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "errors_yield_func.pdf")
    plt.close()


def save_history_plots(config, 
                       losses, 
                       evaluation_epochs, 
                       mean_train_L2_errors,
                       mean_val_L2_errors):
    """
    Saves the training history plots for the state function.
    Args:
        config: configuration object.
        losses: list of training losses over epochs.
        evaluation_epochs: list of epochs at which evaluations were performed.
        mean_train_L2_errors: list of mean L2 errors on the training set over evaluation epochs.
        mean_val_L2_errors: list of mean L2 errors on the validation set over evaluation epochs.
    """
    save_path = Path(config.save_folder)

    plt.figure(figsize=(8,8))
    plt.semilogy(losses)
    plt.xlabel("Training epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.tight_layout()
    plt.title(f"Loss (final = {losses[-1]:.4g})")
    plt.savefig(save_path / "loss.pdf")
    plt.close()

    plt.figure(figsize=(16, 6))
    plt.semilogy(evaluation_epochs, mean_train_L2_errors, label="Mean train L2 error")
    plt.semilogy(evaluation_epochs, mean_val_L2_errors, label="Mean validation L2 error")
    plt.xlabel("Training epochs")
    plt.ylabel("L2 error")
    plt.title(f"Mean L2 error (train = {mean_train_L2_errors[-1]:.4g}, val = {mean_val_L2_errors[-1]:.4g})")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "errors.pdf")
    plt.close()


def save_params_variability_plots(config,
                                  evaluation_epochs,
                                  R_matrix_params_vars,
                                  yield_surf_params_vars):
    """
    Saves the plots for the variability of the R matrix parameters and yield surface parameters over training epochs.
    Args:
        config: configuration object.
        evaluation_epochs: list of epochs at which evaluations were performed.
        R_matrix_params_vars: list of variability measures for the R matrix parameters over evaluation epochs.
        yield_surf_params_vars: list of variability measures for the yield surface parameters over evaluation epochs.
    """

    save_path = Path(config.save_folder)
    fig, axs = plt.subplots(2, 1, figsize=(16, 6))
    axs[0].semilogy(evaluation_epochs, R_matrix_params_vars, label="R matrix params variability")
    axs[0].set_xlabel("Training epochs")
    axs[0].set_ylabel("R matrix params variability")
    axs[0].set_title(f"R matrix params variability (final = {R_matrix_params_vars[-1]:.4g})")
    axs[0].grid()
    axs[1].semilogy(evaluation_epochs, yield_surf_params_vars, label="Yield surface params variability")
    axs[1].set_xlabel("Training epochs")
    axs[1].set_ylabel("Yield surface params variability")
    axs[1].set_title(f"Yield surface params variability (final = {yield_surf_params_vars[-1]:.4g})")
    axs[1].grid()
    plt.tight_layout()
    plt.savefig(save_path / "params_variability.pdf")
    plt.close()


def save_loads_plots(config, epoch, split, L2_errors, forces, targets, input_chars):
    """
    Saves the plots of the predicted and target load curves for each input characteristic in the dataset.
    Args:
        config: configuration object.
        epoch: current training epoch.
        split: dataset split (e.g., "train" or "val").
        L2_errors: list of L2 errors for each input characteristic.
        rel_L2_errors: list of relative L2 errors for each input characteristic.
        forces: predicted forces for each input characteristic.
        targets: target forces for each input characteristic.
        input_chars: list of input characteristics corresponding to the forces and targets.
    """
    
    save_path = Path(config.save_folder)

    for i, input_char in enumerate(input_chars):
        inputs = loading_inputs(config, input_char, targets.shape[1])

        if config.system == "beam":
            fig, axs = plt.subplots(1, 2, figsize=(8, 8))
            fig.suptitle(f"Angle {input_char[1]} (L2 error: {L2_errors[i]:.4g})")
            axs[0].plot(inputs[:,0], targets[i,:,0], label="Target load curve")
            axs[0].plot(inputs[:,0], forces[i,:,0], label="Predicted load curve")
            axs[0].set_xlabel("Mid point strain")
            axs[0].set_ylabel("Normal force [N]")
            axs[0].grid()
            axs[1].plot(inputs[:,1], targets[i,:,1], label="Target load curve")
            axs[1].plot(inputs[:,1], forces[i,:,1], label="Predicted load curve")
            axs[1].set_xlabel("Normalize curvature")
            axs[1].set_ylabel("Normalized moment [N]")
            axs[1].grid()
        elif config.system == "shell":
            fig, axs = plt.subplots(2, 3, figsize=(8, 8))
            fig.suptitle(f"{input_char[1]:.2f} {input_char[2]:.2f} {input_char[3]:.2f} {input_char[4]:.2f} {input_char[5]:.2f} (L2 error: {L2_errors[i]:.4g})")
            axs[0, 0].plot(inputs[:,0], targets[i,:,0], label="Target load curve")
            axs[0, 0].plot(inputs[:,0], forces[i,:,0], label="Predicted load curve")
            axs[0, 0].set_xlabel(r"$\bar{\varepsilon}_{11}$")
            axs[0, 0].set_ylabel(r"$N_{11}$ [N/m]")
            axs[0, 0].grid()
            axs[0, 1].plot(inputs[:,1], targets[i,:,1], label="Target load curve")
            axs[0, 1].plot(inputs[:,1], forces[i,:,1], label="Predicted load curve")
            axs[0, 1].set_xlabel(r"$\bar{\varepsilon}_{22}$")
            axs[0, 1].set_ylabel(r"$N_{22}$ [N/m]")
            axs[0, 1].grid()
            axs[0, 2].plot(inputs[:,2]/jnp.sqrt(2), targets[i,:,2]/jnp.sqrt(2), label="Target load curve")
            axs[0, 2].plot(inputs[:,2]/jnp.sqrt(2), forces[i,:,2]/jnp.sqrt(2), label="Predicted load curve")
            axs[0, 2].set_xlabel(r"$\bar{\varepsilon}_{22}$")
            axs[0, 2].set_ylabel(r"$N_{22}$ [N/m]")
            axs[0, 2].grid()
            axs[1, 0].plot(inputs[:,3], targets[i,:,3], label="Target load curve")
            axs[1, 0].plot(inputs[:,3], forces[i,:,3], label="Predicted load curve")
            axs[1, 0].set_xlabel(r"$\bar{\kappa}_{11}$")
            axs[1, 0].set_ylabel(r"$M_{11}$ [N/m]")
            axs[1, 0].grid()
            axs[1, 1].plot(inputs[:,4], targets[i,:,4], label="Target load curve")
            axs[1, 1].plot(inputs[:,4], forces[i,:,4], label="Predicted load curve")
            axs[1, 1].set_xlabel(r"$\bar{\kappa}_{22}$")
            axs[1, 1].set_ylabel(r"$M_{22}$ [N/m]")
            axs[1, 1].grid()
            axs[1, 2].plot(inputs[:,5]/jnp.sqrt(2), targets[i,:,5]/jnp.sqrt(2), label="Target load curve")
            axs[1, 2].plot(inputs[:,5]/jnp.sqrt(2), forces[i,:,5]/jnp.sqrt(2), label="Predicted load curve")
            axs[1, 2].set_xlabel(r"$\bar{\kappa}_{12}\cdot h_\text{max}$")
            axs[1, 2].set_ylabel(r"$M_{12}/h_\text{max}$ [N/m]")
            axs[1, 2].grid()
        else:
            raise ValueError(f"Unknown system: {config.system}")
        
        plt.tight_layout()
        plt.savefig(save_path / f"{split}_{i}.pdf")
        plt.close()

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parses command-line arguments for the training script.
    Args:
        argv: list of command-line arguments. If None, defaults to sys.argv.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-from",
        type=Path,
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["yield_func", "state_func", "all"],
    )
    parser.add_argument(
        "--pretraining",
        action="store_true",
    )
    args = parser.parse_args(argv)
    return args


def main() -> None:
    """
    Main function to run the training script. 
    """

    args = parse_args()

    if args.resume_from is not None and args.stage is None:
        raise ValueError("Must specify --stage when resuming from a checkpoint.")
    elif args.resume_from is None and args.stage is not None:
        raise ValueError("Must specify --resume-from when specifying --stage.")
    elif args.resume_from is not None and args.stage == "yield_func":
        train_loop_yield_func(resume_from=args.resume_from)
        if args.pretraining:
            config = load_config(Path(args.resume_from).parent)
            train_loop(config)
        else:
            config = load_config(Path(args.resume_from).parent)
            train_loop_state_func(config)
    elif args.resume_from is not None and args.stage == "state_func":
        train_loop_state_func(resume_from=args.resume_from)
    elif args.resume_from is not None and args.stage == "all":
        train_loop(resume_from=args.resume_from)
    else:
        config = Config()
        if config.separated_training:
            train_loop_yield_func(config)
            train_loop_state_func(config)
        else:
            if config.pretraining:
                train_loop_yield_func(config)
                train_loop(config)
            else:
                train_loop(config)


if __name__ == "__main__":
    main()