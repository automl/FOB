# code from: https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/adafactor.py

import math
from typing import Any, Type

import torch
from torch.optim.optimizer import Optimizer

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, PolynomialLR, SequentialLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from engine.configs import OptimizerConfig
from engine.utils import log_info
from engine.parameter_groups import GroupedModel

Eps2 = tuple[float, float]
ParamGroup = dict[str, Any]


class Adafactor(Optimizer):
    """Implements Adafactor algorithm.

    It has been proposed in: `Adafactor: Adaptive Learning Rates with
    Sublinear Memory Cost`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: external learning rate (default: None)
        eps2: regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold: threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate: coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1: coefficient used for computing running averages of gradient
            (default: None)
        weight_decay: weight decay (L2 penalty) (default: 0)
        scale_parameter: if true, learning rate is scaled by root mean square
            of parameter (default: True)
        relative_step: if true, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init: time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Adafactor(model.parameters())
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1804.04235

    Note:
        Reference code: https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py  # noqa
    """

    def __init__(
        self,
        params,
        lr = None,
        eps2: Eps2 = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1 = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
    ):
        if lr is not None and lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            eps2=eps2,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super(Adafactor, self).__init__(params, defaults)

    def _get_lr(self, param_group: ParamGroup, param_state) -> float:
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"]
                if param_group["warmup_init"]
                else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps2"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(
        self, param_group: ParamGroup, param_shape: tuple[int, ...]
    ) -> tuple[bool, bool]:
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(
        self,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    def step(self, closure = None) -> float | None:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adafactor does not support sparse gradients."
                    )

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(
                    group, grad_shape
                )
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format
                        )
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad_shape[:-1]
                        ).type_as(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).type_as(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format
                        )

                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps2"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t
                    )

                    # Approximation of exponential moving average of square
                    # of gradient
                    self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col, update
                    )
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    torch.rsqrt(exp_avg_sq, out=update).mul_(grad)

                update.div_(
                    max(1.0, self._rms(update) / group["clip_threshold"])
                )
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(
                        update, alpha=1 - group["beta1"]
                    )
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * lr)

                p.data.add_(-update)

        return loss

def warmup_split(max_steps: int, warmup_factor: float) -> tuple[int, int]:
    warmup_steps = int(math.ceil(max_steps * warmup_factor))
    return warmup_steps, max(max_steps - warmup_steps, 1)


def linear_warmup(
        optimizer,
        max_steps: int,
        warmup_steps: int,
        scheduler: Type[CosineAnnealingLR] | Type[PolynomialLR],
        scheduler_kwargs: dict[str, Any]
    ) -> SequentialLR | CosineAnnealingLR | PolynomialLR:
    if max_steps < 1:
        raise ValueError("max steps should be at least 1!")
    if warmup_steps == 0:
        log_info(f"warmup = 0: using {scheduler} only")
        return scheduler(optimizer, **scheduler_kwargs)
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    actual_scheduler = scheduler(optimizer, **scheduler_kwargs)
    return SequentialLR(
        optimizer, schedulers=[warmup, actual_scheduler], milestones=[warmup_steps])


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    weight_decay = config.weight_decay
    parameter_groups = model.grouped_parameters(lr=lr, weight_decay=weight_decay)
    min_lr = config.eta_min_factor * lr
    optimizer = Adafactor(
        parameter_groups,
        lr=lr,
        eps2=(config.eps1, config.eps2),
        clip_threshold=config.clipping_threshold,
        decay_rate=config.decay_rate,
        beta1=(1.0 - config.one_minus_beta1),
        weight_decay=weight_decay,
        scale_parameter=True,
        relative_step=False,
        warmup_init=False
    )
    warmup_steps, scheduler_steps = warmup_split(config.max_steps, config.warmup_factor)
    if config.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR
        scheduler_kwargs = dict(
            T_max=scheduler_steps,
            eta_min=min_lr
        )
    elif config.lr_scheduler == "poly":
        scheduler = PolynomialLR
        scheduler_kwargs = dict(
            power=config.lr_power,
            total_iters=scheduler_steps
        )
    else:
        raise ValueError(f"unknown lr_scheduler: {config.lr_scheduler}")
    lr_scheduler = linear_warmup(
        optimizer,
        max_steps=config.max_steps,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": config.lr_interval
        }
    }
