# code from: https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/adafactor.py

import math
from typing import Any

import torch
from torch.optim.optimizer import Optimizer

from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.optimizers.lr_schedulers import get_lr_scheduler



class WAdam(Optimizer):
    """Implements Welfordized Adam.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: external learning rate (default: None)
        beta1: coefficient used for computing running averages of gradient
            (default: None)
        beta2: coefficient used for computing running averages of gradient
            (default: None)
        epsilon: a small constant for numerical stability (default: 1e-7)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2 = None,
        epsilon = 1e-7,
    ):
        if lr is not None and lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if beta1 < 0.0:
            raise ValueError(
                "Invalid beta1 value: {}".format(beta1)
            )
        if beta2 is None:
            beta2 = beta1
        if beta2 < 0.0:
            raise ValueError(
                "Invalid beta2 value: {}".format(beta2)
            )
        if epsilon < 0.0:
            raise ValueError(
                "Invalid epsilon value: {}".format(epsilon)
            )

        hyperparams = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        super(WAdam, self).__init__(params, hyperparams)

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
                g = p.grad.data
                if g.is_sparse:
                    raise RuntimeError(
                        "Wadam does not support sparse gradients."
                    )

                state = self.state[p]
                grad_shape = g.shape

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["running_mean"] = torch.zeros(
                        grad_shape,
                    ).type_as(g)
                    state["running_variance"] = torch.zeros(
                        grad_shape,
                    ).type_as(g)

                state['step'] += 1
                step = state['step']
                step = state['step']
                m = state['running_mean']
                v = state['running_variance']

                lr = group['lr']
                beta1 = group['beta1']
                beta2 = group['beta2']
                eps = group['epsilon']

                delta = g - m
                m.add_((1. - beta1) * delta)
                v.add_(
                    (beta2 - 1.) * v + (1. - beta2) * delta * (g - m)
                )

                #Debiased step size
                beta1_power = beta1 ** step
                beta2_power = beta2 ** step
                alpha = lr * math.sqrt(1. - beta2_power) / (1. - beta1_power)
                print(f'learning rate: {lr}')
                p.data.sub_((m * alpha) / (torch.sqrt(v) + eps))

        return loss

def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    beta1 = config.beta1
    beta2 = config.beta2
    epsilon = config.epsilon
    optimizer = WAdam(
        model.grouped_parameters(lr=lr), 
        lr=lr, 
        beta1=beta1, 
        beta2=beta2,
        epsilon=epsilon,
    )
    lr_scheduler = get_lr_scheduler(optimizer, config)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": config.lr_scheduler.interval
        }
    }


if __name__=='__main__':
    params = torch.tensor([0., 0.], requires_grad=True)
    maxiter = 1000
    opt = WAdam([params], 1e-3, 0.9, 0.9, 1e-3)
    mc_samples = 30
    def loss_fn():
        opt.zero_grad()
        loc,scale = params
        scale = torch.exp(scale)
        q = torch.distributions.Normal(loc, scale)
        p = torch.distributions.Normal(10., 1.)
        z = q.rsample((mc_samples,))
        loss = torch.mean(q.log_prob(z) - p.log_prob(z))
        loss.backward()
        return loss

    from tqdm import trange
    loss = []
    for i in trange(maxiter):
        loss.append(float(opt.step(loss_fn)))


