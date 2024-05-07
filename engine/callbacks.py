import math
import time
from typing import Optional
import deepspeed
import torch
import lightning.pytorch as pl
from lightning import Callback, Trainer, LightningModule
from lightning_utilities.core.rank_zero import rank_zero_only
from engine.utils import log_debug, log_info, log_warn, seconds_to_str


class RestrictTrainEpochs(Callback):
    """Counts number of epochs since start of training and stops if max_epochs is reached."""
    def __init__(self, max_epochs: int):
        super().__init__()
        self.max_epochs = max_epochs
        self.epochs = 0
        self.skip_first = False

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        log_debug(f"Training for {self.max_epochs} epochs...")
        self.epochs = 0
        trainer.should_stop = False

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.skip_first:
            self.skip_first = False
        else:
            self.epochs += 1
        log_debug(f"Epoch {self.epochs}/{self.max_epochs}")
        # TODO: test for DDP, do we need 'trainer.strategy.reduce_boolean_decision'?
        if self.epochs >= self.max_epochs:
            log_debug(f"Stopping training after {self.epochs} epochs")
            trainer.should_stop = True

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint):
        # checkpoint loads the model at the end of the epoch, so we do not count the first epoch
        self.skip_first = True


class PrintEpochWithTime(Callback):
    def __init__(self, active: bool = True):
        super().__init__()
        self.active: bool = active
        self.time: dict[str, Optional[float]]
        self.reset_time()

    def reset_time(self):
        self.time = {
            "train_start": None,
            "val_start": None,
            "val_end": None
        }

    @rank_zero_only
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        if self.active:
            self.time["train_start"] = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # need to print here since train epoch ends after validation is done
        if self.active and all(v is not None for v in self.time.values()):
            max_epochs = pl_module.config.max_epochs
            train_time = math.ceil(time.time() - self.time["train_start"])  # type: ignore
            val_time = math.ceil(self.time["val_end"] - self.time["val_start"])  # type: ignore
            log_info(f"Finished training epoch {trainer.current_epoch + 1} of {max_epochs}. Time spent: training: {seconds_to_str(train_time - val_time)}, validation: {seconds_to_str(val_time)}, total: {seconds_to_str(train_time)}.")
            self.reset_time()

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        if self.active:
            self.time["val_start"] = time.time()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.active:
            self.time["val_end"] = time.time()


class LogParamsAndGrads(Callback):
    def __init__(self, log_gradient: bool, log_params: bool, log_quantiles: bool, log_every_n_steps: int):
        super().__init__()
        self.log_gradient = log_gradient
        self.log_params = log_params
        self.log_quantiles = log_quantiles
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer):

        if trainer.global_step % self.log_every_n_steps == 0 and (self.log_params or self.log_gradient):

            q = torch.arange(.25, 1, .25).round(decimals=2).to(trainer.model.device)
            stats = {}
            for k, v in pl_module.named_parameters():
                if self.log_params:
                    if trainer.global_rank == 0:
                        v_detached = v.detach()

                        if torch.isnan(v_detached).sum() > 0:
                            log_warn(f"# NaN in param {k}")
                        if torch.isinf(v_detached).sum() > 0:
                            log_warn(f"# Inf in param {k}")

                        stats[f"param/{k}/mean"] = v_detached.mean().item()
                        if v_detached.shape[0] > 1:
                            stats[f"param/{k}/std"] = v_detached.std().item()
                            stats[f"param/{k}/min"] = v_detached.min().item()
                            stats[f"param/{k}/max"] = v_detached.max().item()
                            stats[f"param/{k}/abs_mean"] = v_detached.abs().mean().item()
                            stats[f"param/{k}/abs_std"] = v_detached.abs().std().item()
                            stats[f"param/{k}/l2m"] = (v_detached**2).mean().item()
                            stats[f"param/{k}/l2s"] = (v_detached**2).sum().item()

                            if self.log_quantiles and v_detached.size().numel() < 10000000:
                                deciles = torch.quantile(v_detached.float(), q, interpolation='linear')
                                for q_idx, d_val in enumerate(deciles):
                                    stats[f"param/{k}/quantile-{q[q_idx]}"] = d_val.item()

                if self.log_gradient and v.requires_grad:

                    if trainer.num_devices > 1:
                        grad_data = deepspeed.utils.safe_get_full_grad(v)
                    else:
                        grad_data = v.grad

                    if grad_data is not None and trainer.global_rank == 0:
                        if torch.isnan(grad_data).sum() > 0:
                            log_warn(f"# NaN in grad {k}")
                        if torch.isinf(grad_data).sum() > 0:
                            log_warn(f"# Inf in grad {k}")

                        if torch.isnan(grad_data).sum() > 0 or torch.isinf(grad_data).sum() > 0:
                            stats[f"grad/{k}/mean"] = -10
                            if len(grad_data.shape) > 1 or grad_data.shape[0] > 1:
                                stats[f"grad/{k}/std"] = -10
                                stats[f"grad/{k}/min"] = -10
                                stats[f"grad/{k}/max"] = -10
                                stats[f"grad/{k}/abs_mean"] = -10
                                stats[f"grad/{k}/abs_std"] = -10
                                if self.log_quantiles and grad_data.size().numel() < 10000000:
                                    for q_idx, _ in enumerate(q):
                                        stats[f"param/{k}/quantile-{q[q_idx]}"] = -10

                        stats[f"grad/{k}/mean"] = grad_data.mean().item()
                        if len(grad_data.shape) > 1 or grad_data.shape[0] > 1:
                            stats[f"grad/{k}/std"] = grad_data.std().item()
                            stats[f"grad/{k}/min"] = grad_data.min().item()
                            stats[f"grad/{k}/max"] = grad_data.max().item()
                            stats[f"grad/{k}/abs_mean"] = grad_data.abs().mean().item()
                            stats[f"grad/{k}/mean"] = grad_data.mean().item()
                            stats[f"grad/{k}/abs_std"] = grad_data.abs().std().item()
                            stats[f"grad/{k}/std"] = grad_data.std().item()

                            if self.log_quantiles and grad_data.size().numel() < 10000000:
                                deciles = torch.quantile(grad_data.float(), q, interpolation='linear')
                                for q_idx, d_val in enumerate(deciles):
                                    stats[f"grad/{k}/quantile-{q[q_idx]}"] = d_val.item()

            if trainer.loggers is not None and trainer.global_rank == 0:
                for logger in trainer.loggers:
                    logger.log_metrics(stats, step=trainer.global_step)
